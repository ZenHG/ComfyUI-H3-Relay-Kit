# -*- coding: utf-8 -*-
"""H3 Relay Kit · 核心算法层（纯张量，零 GPU、零模型、可离线单测）

【这是什么】
MiniMax-H3 多段续接的 **latent 桥**：把上一段的 AV latent 切出尾段，
以 `minimax_keyframes`（多块，按 `resolved_frame_index` 排位）
+ `minimax_refs`（音频）注入本段的 conditioning。

【与像素续接的区别（本包存在的理由）】
  像素续接：上一段 mp4 → 解码帧 → VAE 重编码 → 单块 keyframe @frame0
            —— 多一次 VAE 往返，有量化损失，且曝光可能漂移。
  latent 桥：上一段 AV latent → 直接切尾段 → 多块 keyframe 按位置
            —— 零重编码，与采样时的 latent 逐位同源。

二者走的是**同一个 ComfyUI 原生协议**（`minimax_keyframes`），
故可互换；本包提供后者，并保持与前者完全兼容的键结构。

【协议出处（本机实测，非推测）】
  comfy/ldm/minimax/model.py:376
      cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]
  comfy/model_base.py:2186-2196
      keyframes = kwargs.get("minimax_keyframes")   → payload["keyframes"]
      refs      = kwargs.get("minimax_refs")        → payload["refs"]
  → 原生支持任意位置的 keyframe 锚，无需任何 monkey patch。

【帧 / latent 网格】
  H3 VAE 的时序跨度为 (1,4,4,4,4)：每 5 个 latent token 覆盖 17 像素帧。
  因此**只有落在网格上的窗口**才能被整步切出，合法窗口（GUIDE_RUNS）：
      1, 5, 22, 39, 56, 73, 90, 107, 124 ...
  22 帧 = 7 个 latent token；39 帧 = 12 个 token。
  不在网格上 → raise（而不是悄悄挪一格，那会渲染出位移的接缝）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------- 网格常量
# 与 H3 VAE 的时序跨度一致。改这里等于改 H3 的 VAE，必须同步。
FRAME_PER_TOKEN: Tuple[int, ...] = (1, 4, 4, 4, 4)
FPS: float = 24.0
FRAME_RESCALE: float = 5.0 / 3.0   # 像素帧 → 音频 latent 步的换算
AUDIO_HZ: float = 40.0             # 音频 latent 的采样率（步/秒）

# 合法窗口，降序。必须与 FRAME_PER_TOKEN 自洽（见 self_check）。
GUIDE_RUNS: Tuple[int, ...] = (124, 107, 90, 73, 56, 39, 22, 5, 1)

# Apt_Preset 在 latent 里留下的导出尾段键（有则优先复用，避免二次切片）
KEY_EXPORT_TAIL_VIDEO = "apt_h3_export_tail_latent"
KEY_EXPORT_TAIL_AUDIO = "apt_h3_export_tail_audio_latent"
KEY_EXPORT_FRAMES = "apt_h3_export_context_frames"


# ---------------------------------------------------------------- 网格工具
def pixel_frames(latent_t: int) -> int:
    """latent token 数 → 覆盖的像素帧数。"""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def step_offsets(latent_t: int) -> List[int]:
    """每个 latent token 对应的**起始像素帧位置**。"""
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def steps_for_frames(n: int) -> Optional[int]:
    """像素帧数 → 整步的 latent token 数；不在网格上返回 None。"""
    n = int(n)
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def snap_guide_run(n: int) -> int:
    """向下吸附到最近的合法窗口（供 UI 侧提示用；核心路径仍会硬校验）。"""
    n = int(n)
    for g in GUIDE_RUNS:
        if g <= n:
            return g
    return 0


def self_check() -> List[str]:
    """自洽性检查：GUIDE_RUNS 必须全部落在网格上。返回问题列表（空=通过）。"""
    bad = []
    for g in GUIDE_RUNS:
        if steps_for_frames(g) is None:
            bad.append(f"GUIDE_RUNS 含非网格值 {g}（pixel_frames 无法整步覆盖）")
    return bad


# ---------------------------------------------------------------- latent 取流
def streams_from_latent(latent: Any) -> List[torch.Tensor]:
    """从 LATENT 取 [video, audio, ...]。

    支持三种形态：
      - ``NestedTensor``（H3 AV 联合 latent，本产线常态）→ 按 ``.tensors`` 拆流
      - list/tuple（已拆开的流）
      - 单个 ``torch.Tensor``（video-only latent）→ 当作**一条**流

    ⚠️ 不要用 ``hasattr(samples, "unbind")`` 来判定"是不是 NestedTensor" ——
    **任何 torch.Tensor 都有 unbind**，普通 ``[B,C,T,H,W]`` 会被沿 batch 维拆开，
    B=1 时碰巧看不出错、B>1 时静默产出垃圾流。故这里显式按类型分支。
    """
    if latent is None:
        raise ValueError("latent 为空：续接需要一段真实的 H3 AV latent。")
    samples = latent["samples"] if isinstance(latent, dict) else latent
    nested = getattr(samples, "tensors", None)
    if nested is not None:
        parts = list(nested)
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    elif torch.is_tensor(samples):
        parts = [samples]
    else:
        raise ValueError(
            "期望 H3 的 AV latent（nested 视频/音频对），得到 %r。\n"
            "    接续接错的常见原因：把 video-only latent 接到了 context_latent。" % type(samples)
        )
    parts = [t for t in parts if torch.is_tensor(t)]
    if not parts:
        raise ValueError("AV latent 里没有可用的张量流。")
    return parts


def video_from_latent(latent: Any) -> torch.Tensor:
    """取视频流，规范成 [B,C,T,H,W]。"""
    v = streams_from_latent(latent)[0]
    if v.ndim == 4:
        v = v.unsqueeze(0)
    if v.ndim != 5:
        raise ValueError(
            "期望视频 latent 形状 [B,C,T,H,W]，得到 %s。" % (tuple(v.shape),)
        )
    return v


def audio_from_latent(latent: Any) -> torch.Tensor:
    """取音频流，规范成 [B,C,2,T]。"""
    parts = streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "context_latent 没有音频流。\n"
            "    续接需要采样器的 AV 输出（视频+音频同源），不是纯视频 latent。"
        )
    a = parts[1]
    if a.ndim == 3:
        a = a.unsqueeze(0)
    if a.ndim != 4:
        raise ValueError("期望音频 latent 形状 [B,C,2,T]，得到 %s。" % (tuple(a.shape),))
    return a


# ---------------------------------------------------------------- 尾段切片
def video_tail_from_latent(
    latent: Any, frames: int
) -> Tuple[List[torch.Tensor], List[int], int]:
    """从 AV latent 切出 ``frames`` 帧的**视频尾段**，切成逐 token 的块。

    返回 ``(blocks, offsets, covered)``：
      - ``blocks[k]`` 是第 k 个 latent token（[1,C,1,H,W]）
      - ``offsets[k]`` 是该 token 的起始像素帧位置
      - ``covered`` 是实际覆盖帧数（等于 frames，否则 raise）

    三条硬约束（违反即 raise，绝不静默降级）：
      1. ``frames`` 必须落在网格上；
      2. 尾段不得长于 latent 本身；
      3. 尾段起始必须落在 5-token 周期边界，否则各 token 的帧跨度
         会和写入的位置对不上 → 渲染出**整体位移的接缝**。
    """
    frames = int(frames)

    # 优先复用外层（Apt 链）已经写好的导出尾段，省一次切片
    exported = latent.get(KEY_EXPORT_TAIL_VIDEO) if isinstance(latent, dict) else None
    exported_frames = int(
        latent.get(KEY_EXPORT_FRAMES, 0) if isinstance(latent, dict) else 0
    )
    if exported is not None and exported_frames and frames == exported_frames:
        if exported.ndim == 4:
            exported = exported.unsqueeze(0)
        steps = steps_for_frames(frames)
        if exported.ndim != 5 or int(exported.shape[2]) != steps:
            raise ValueError(
                "latent 内附的导出尾段与 %d 帧的 H3 网格不符（形状 %s，期望 %d 步）。"
                % (frames, tuple(exported.shape), steps)
            )
        blocks = [exported[:1, :, k:k + 1].clone() for k in range(steps)]
        return blocks, step_offsets(steps), frames

    video = video_from_latent(latent)
    total = int(video.shape[2])
    steps = steps_for_frames(frames)
    if steps is None:
        raise ValueError(
            "%d 帧不是 H3 latent 的整步窗口，无法从 latent 切片。\n"
            "    合法窗口：%s。\n"
            "    若确实要用这个帧数，请改用像素路径（context_frames）而不是 context_latent。"
            % (frames, ", ".join(str(g) for g in GUIDE_RUNS if g > 1))
        )
    if steps > total:
        raise ValueError(
            "需要 %d 个 latent 步（%d 帧），但 context_latent 只有 %d 步。\n"
            "    上一段太短，或分辨率/时长与这一段不匹配。"
            % (steps, frames, total)
        )
    start = total - steps
    if start % 5 != 0:
        raise RuntimeError(
            "%d 步的尾段在 %d 步的 latent 里起始于周期位置 %d（非 0），"
            "各 token 的帧跨度会与写入位置错位 → 接缝整体位移。\n"
            "    通常说明段长不匹配；调整段长使 (总步数 - %d) 是 5 的倍数。"
            % (steps, total, start % 5, steps)
        )
    covered = pixel_frames(steps)
    if covered != frames:
        raise RuntimeError(
            "%d 步覆盖 %d 帧，期望 %d 帧（网格自洽性被破坏）。" % (steps, covered, frames)
        )
    blocks = [video[:1, :, start + k:start + k + 1].clone() for k in range(steps)]
    return blocks, step_offsets(steps), covered


def audio_tail_from_latent(
    latent: Any, a_frames: int, src_total_frames: int
) -> Tuple[torch.Tensor, int, float, float, bool]:
    """从 AV latent 切出 ``a_frames`` 帧对应的**音频尾段**。

    ``src_total_frames`` 是**源段（latent）的总像素帧数**——H3 的音频 tick 数按
    总帧数四舍五入生成，外溢偏差只有对着总帧数量才有意义（对着尾窗量恒为巨值）。

    返回 ``(tail, ref_audio_t, overhang, raw_steps, grid_off)``。
    非 40Hz 网格整步的 ``a_frames`` 会向上拓宽到最近整步（``ref_audio_t``），
    ``raw_steps`` 是换算的理论步数，供上层留注记。
    ``grid_off`` 为 True 表示音频栅格偏差超出半整步（输入段可能非标准网格），
    此时 ``overhang`` 按 0 处理，告警文案由上层写进 notes。
    """
    a_frames = int(a_frames)

    exported = latent.get(KEY_EXPORT_TAIL_AUDIO) if isinstance(latent, dict) else None
    if exported is not None:
        if exported.ndim == 3:
            exported = exported.unsqueeze(0)
        if exported.ndim != 4:
            raise ValueError("latent 内附的导出音频尾段形状非法。")
        n_t = int(exported.shape[-1])
        return exported[:1].clone(), n_t, 0.0, float(n_t), False

    audio = audio_from_latent(latent)
    total_t = int(audio.shape[-1])
    overhang = total_t - FRAME_RESCALE * int(src_total_frames)
    grid_off = not (-0.5 < overhang < 0.5)
    if grid_off:
        # H3 把音频栅格四舍五入到最近的步；偏差过大只可能是输入不对，
        # 这里按无外溢处理，由 grid_off 标志让上层决定是否告警。
        overhang = 0.0

    raw_steps = a_frames / float(FPS) * AUDIO_HZ
    # 非 40Hz 网格整步的值一律**向上拓宽**到最近整步：
    # 音频窗的作用是给模型"已经播过的声音"当上下文，多带半步是安全的，
    # 截短半步则可能丢掉节拍点。换算误差只往"多带"方向偏。
    rt = int(math.ceil(raw_steps - 1e-9))
    if rt > total_t:
        rt = total_t
    if rt < 1:
        raise ValueError("音频窗口为空（%d 帧换算后不足一步）。" % a_frames)
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang), raw_steps, grid_off


# ---------------------------------------------------------------- 续接计划
@dataclass
class RelayPlan:
    """一次续接的完整计划（可打印、可序列化，便于留痕排障）。"""

    applied: bool = False
    span: int = 0                      # 被钉住的像素帧数
    steps: int = 0                     # 被钉住的 latent token 数
    trim: int = 0                      # 下游应裁掉的首部**像素帧**数 = span + settle
    settle: int = 0                    # 其中属于「沉降区」的帧数（不受 5+17k 网格约束）
    indices: List[int] = field(default_factory=list)
    keyframes: List[Dict[str, Any]] = field(default_factory=list)
    audio_ref: Optional[Dict[str, Any]] = None
    clipped_channels: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.applied:
            return "未应用（context_latent 未接）"
        line = (
            "钉住 %d 帧 / %d 步，锚位 %d..%d，裁首 %d 帧（含沉降 %d），音频 %s"
            % (
                self.span, self.steps,
                self.indices[0] if self.indices else -1,
                self.indices[-1] if self.indices else -1,
                self.trim,
                self.settle,
                ("%d 步" % self.audio_ref["ref_audio_t"]) if self.audio_ref else "关",
            )
        )
        return line


def plan_relay(
    latent: Any,
    context_latent: Any,
    trim_frames: int = 22,
    audio_frames: Optional[int] = None,
    settle_frames: int = 0,
) -> RelayPlan:
    """生成续接计划。

    参数
      latent          本段的目标 latent（提供分辨率 / 步数 / 帧数）
      context_latent  上一段的 AV latent（提供被钉住的尾段）
      trim_frames     钉住的像素帧数，必须在 GUIDE_RUNS 上
      audio_frames    音频钉住窗口（像素帧口径）；默认与视频同窗
      settle_frames   沉降帧数（默认 0 = 只裁钉住区）。钉住区之后模型还会先
                      **复现**上一段若干帧才切到本段 prompt，那几帧一并裁掉
                      才能保证拼接处不跳变、上段文字不串入。不受 5+17k 网格
                      约束 —— 裁的是 decode 之后的像素帧，不动 latent 相位。
                      （UI 上的自动检测由 H3RelayTrimAV 做，见 detect_settle）

    分辨率必须一致 —— latent 无法缩放，不一致只能重跑上一段或从本段重启链。
    """
    plan = RelayPlan()
    if context_latent is None:
        plan.notes.append("未接 context_latent：本段不续接（按独立段处理）。")
        return plan

    trim_frames = int(trim_frames)
    settle_frames = max(0, int(settle_frames))
    dst = video_from_latent(latent)
    src = video_from_latent(context_latent)
    w, h = int(dst.shape[4]) * 16, int(dst.shape[3]) * 16
    sw, sh = int(src.shape[4]) * 16, int(src.shape[3]) * 16
    if (sw, sh) != (w, h):
        raise ValueError(
            "context_latent 是 %dx%d，本段是 %dx%d。latent 无法缩放，"
            "续接要求两段同分辨率。\n"
            "    解决：让上一段用同分辨率重跑，或把链从这里重启。"
            % (sw, sh, w, h)
        )
    if int(src.shape[1]) != int(dst.shape[1]):
        raise ValueError(
            "context_latent 有 %d 个通道，本段有 %d 个 —— 不是同一底模产出的 H3 视频 latent。"
            % (int(src.shape[1]), int(dst.shape[1]))
        )

    frame_count = pixel_frames(int(dst.shape[2]))
    if trim_frames not in GUIDE_RUNS:
        # 不静默吸附：改了帧数就是改了续接窗口，必须让你知道。
        near = snap_guide_run(trim_frames)
        hint = ("最接近的合法窗口是 %d。" % near) if near else "没有比它更小的合法窗口。"
        raise ValueError(
            "%d 帧不是 H3 的整步续接窗口。\n"
            "    合法值：%s（= 5 + 17k，配 17k+5 帧段长时尾段起点正好落在 5 步周期边界）。\n"
            "    %s"
            % (trim_frames, ", ".join(str(g) for g in sorted(GUIDE_RUNS) if g >= 5), hint)
        )
    if trim_frames >= frame_count:
        raise ValueError(
            "钉住 %d 帧、本段只有 %d 帧 —— 没有新内容可生成。\n"
            "    缩短窗口或加长本段。" % (trim_frames, frame_count)
        )
    if trim_frames + settle_frames >= frame_count:
        raise ValueError(
            "钉住 %d 帧 + 沉降 %d 帧 = %d 帧 ≥ 本段 %d 帧 —— 裁完就没画面了。\n"
            "    调小沉降帧数，或加长本段。"
            % (trim_frames, settle_frames, trim_frames + settle_frames, frame_count)
        )

    blocks, offsets, covered = video_tail_from_latent(context_latent, trim_frames)
    plan.span = covered
    plan.steps = len(blocks)
    plan.indices = list(offsets)
    plan.settle = settle_frames
    # pin 受 5+17k 网格约束（上面已硬校验）；settle 裁的是像素帧，不影响 latent 相位。
    plan.trim = covered + settle_frames
    if settle_frames:
        plan.notes.append(
            "沉降 %d 帧：钉住区之后模型还会先复现若干帧才切到本段 prompt，"
            "这几帧一并裁掉，避免拼接处跳变与上段文字串入。" % settle_frames
        )

    plan.keyframes = [
        {"resolved_frame_index": int(p), "latent": blk}
        for p, blk in zip(plan.indices, blocks)
    ]

    a_frames = int(audio_frames) if audio_frames else trim_frames
    src_total_frames = pixel_frames(int(src.shape[2]))
    tail, rt, overhang, raw_steps, grid_off = audio_tail_from_latent(
        context_latent, a_frames, src_total_frames)
    if grid_off:
        plan.notes.append(
            "⚠ 音频栅格与视频帧数偏差超出半整步（上段 %d 帧，音频栅格换算后与之对不上），"
            "已按无外溢处理——输入段可能不是标准 H3 网格，请检查上一段来源。"
            % src_total_frames
        )
    if rt > raw_steps + 1e-6:
        plan.notes.append(
            "音频窗 %d 帧换算 %.2f 步，已拓宽到 %d 整步（宁多带、不截短）。"
            % (a_frames, raw_steps, rt)
        )
    dst_audio_t = None
    try:
        dst_audio_t = int(audio_from_latent(latent).shape[-1])
    except ValueError:
        dst_audio_t = None
    if dst_audio_t is not None and rt > dst_audio_t:
        plan.notes.append(
            "音频窗口 %d 步超过本段音频栅格 %d 步，已截断。" % (rt, dst_audio_t)
        )
        tail = tail[..., :dst_audio_t].clone()
        rt = dst_audio_t
    plan.audio_ref = {"kind": "audio", "ref_audio_t": int(rt), "audio_latent": tail}
    if overhang:
        plan.notes.append("音频栅格外溢 %.3f 步（已在放置时对齐）。" % overhang)

    plan.applied = True
    return plan


def apply_relay(conditioning, plan: RelayPlan):
    """把计划注入 conditioning：keyframes 合并 + 音频 ref 追加。

    与上游既有 keyframes（如 first/last 帧锚）**合并而非替换**；
    落在钉住区内的旧锚会被丢弃（它们与钉住区冲突）。
    """
    import node_helpers

    if not plan.applied:
        return conditioning

    head_end = plan.span
    out, dropped = [], []
    for emb, extra in conditioning:
        d = dict(extra)
        prior = list(d.get("minimax_keyframes") or [])
        kept = []
        for kf in prior:
            pos = int(kf.get("resolved_frame_index", 0))
            if pos < head_end:
                dropped.append(pos)
                continue
            kept.append(dict(kf))
        d["minimax_keyframes"] = kept + [dict(k) for k in plan.keyframes]
        out.append([emb, d])

    if dropped:
        plan.notes.append(
            "丢弃 %d 个落在钉住区（0..%d）内的旧锚，避免与续接区冲突。"
            % (len(set(dropped)), head_end - 1)
        )

    if plan.audio_ref is not None:
        out = node_helpers.conditioning_set_values(
            out, {"minimax_refs": [plan.audio_ref]}, append=True
        )
    return out


# ---------------------------------------------------------------- AV latent 存取
# 磁盘格式：一个 safetensors，内含 streams.video / streams.audio + metadata。
# NestedTensor 无法直接 safetensors 序列化，故按流拆开存。
_LATENT_META_KEY = "relay_kit_meta"


def save_av_latent(latent: Any, path: str, note: str = "") -> str:
    """把 H3 的 AV latent 落盘，供下一段当 context_latent 读回。"""
    from safetensors.torch import save_file

    parts = streams_from_latent(latent)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tensors: Dict[str, torch.Tensor] = {}
    names = []
    for i, t in enumerate(parts):
        name = "video" if i == 0 else ("audio" if i == 1 else "stream_%d" % i)
        names.append(name)
        # safetensors 要求连续内存。detach() 已脱离 autograd 共享（与旧版 clone 的
        # 安全性等价），跨设备时 .cpu() 本来就物化新张量——不再额外 clone，
        # 落盘瞬间的 CPU 峰值从 2-3 倍降到 1 倍。
        tc = t.detach()
        if tc.device.type != "cpu":
            tc = tc.cpu()
        tensors["streams." + name] = tc.contiguous()
    meta = {
        "format": 1,
        "streams": names,
        "shapes": [list(t.shape) for t in parts],
        "note": str(note),
    }
    # UTF-8 字节流：note 含中文时 ord(c) 会超 uint8 直接崩，必须先 encode
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    tensors[_LATENT_META_KEY] = torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8)
    # 原子写：先写同目录 .tmp 再替换，中途崩溃不会留下截断的 safetensors
    tmp = path + ".tmp"
    try:
        save_file(tensors, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def load_av_latent(path: str):
    """读回落盘的 AV latent，重建 NestedTensor。"""
    from safetensors.torch import load_file

    try:
        import comfy.nested_tensor as _nt
    except Exception:  # pragma: no cover - ComfyUI 运行时一定可用
        _nt = None

    if not os.path.isfile(path):
        raise FileNotFoundError("读不到续接 latent：%s" % path)
    raw = load_file(path)
    meta_t = raw.pop(_LATENT_META_KEY, None)
    if meta_t is None:
        raise ValueError("%s 不是本工具写的 AV latent（缺少元数据）。" % path)
    meta = json.loads(bytes(meta_t.tolist()).decode("utf-8"))
    parts = [raw["streams." + n] for n in meta["streams"]]
    if _nt is not None:
        samples = _nt.NestedTensor(parts)
    else:  # pragma: no cover
        samples = parts
    return {"samples": samples}


def trim_head_frames(images: torch.Tensor, frames: int) -> torch.Tensor:
    """裁掉 IMAGE 的前 ``frames`` 帧（沿第 0 维）。

    续接段的前 ``trim_frames`` 帧是对上一段尾部的**重生成**（实测逐帧 MAE ≈ 6/255），
    它们是钉住区的产物，不属于新内容 —— 不裁就会在拼接处看到约 0.9s 重播。
    """
    frames = int(frames)
    if frames <= 0:
        return images
    n = int(images.shape[0])
    if frames >= n:
        raise ValueError(
            "要裁 %d 帧，但本段只有 %d 帧 —— 裁完就没画面了。\n"
            "    检查 trim_frames 是否误填成整段长度。" % (frames, n)
        )
    return images[frames:]


def trim_audio_head(audio: Any, frames: int, fps: float = FPS) -> Any:
    """把 AUDIO 的前 ``frames`` 帧对应的采样点裁掉（与视频同裁，保住 A/V 同步）。"""
    frames = int(frames)
    if frames <= 0 or audio is None:
        return audio
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    n = int(round(frames / float(fps) * sr))
    total = int(wf.shape[-1])
    if n >= total:
        raise ValueError(
            "要裁 %d 帧（%d 个采样点），但音频只有 %d 点。"
            % (frames, n, total)
        )
    return {"waveform": wf[..., n:], "sample_rate": sr}


# ---------------------------------------------------------------- 接缝自检
# 实测（2026-09-11）：续接段的「钉住区 → 新内容」切换**不总落在 trim 值上**。
# 73 帧段实测切换点在原第 22→23 帧之间（MAE 6.5 → 102.3），
# 而 trim=22 恰好把切换点前的那一帧留在裁剪后的第 0 帧 → 首帧突变。
# 107 帧段同一 trim 值则无此现象（切换点被完整裁掉）。
# ⇒ trim 值必须按**本段的实际切换点**定，不能写死。
#
# 落点：切换点是**逐段不同的随机变量**，所以它不该是用户填的配置，而是
# **观测出来的量**。裁节点手里就是完整 decode 序列（含钉住区），真实切换点
# 此刻就在手上 —— `detect_settle` 就在那里量，量不出就退回只裁钉住区。
JUMP_RATIO: float = 4.0       # 首帧差 / 段内基线 的报警阈值
# 基线过小时的保护下限——**随数据量纲缩放**（v0.3.2）：ComfyUI decode 张量是 0-1，
# 旧写死 1.0（0-255 量纲）会把 0-1 数据的帧差路径整个抬死（基线 0.0076 被抬到 1.0，
# 阈值 4.0 永不可达 → 帧差法在产线上从未生效过）。0.004 ≈ 1/255 满量程，
# 对 0-255 的旧测试行为不变（0.004×250 ≈ 1.0）。
BASELINE_FLOOR_REL: float = 0.004
MAX_SETTLE: int = 12          # 沉降帧上限（≈0.5 s）。自动检测不会超过它
# 裁后仍见突变时，只有位置在这么靠前才值得动刀：
# 裁后下标 j 表示"还差 j+1 帧沉降"，所以上限是 MAX_SETTLE-1。
ADVISE_WITHIN: int = MAX_SETTLE - 1
# —— 模糊型沉降（v0.3.1）：切换不是硬跳而是「重绘发虚」——帧差法看不见，改看高频能量。——
# 度量用 **Laplacian 响应的均方**（≈方差）：模糊先杀高对比边缘，平方放大这一损失。
# mean-abs 版实测钝 3 倍（HD mild 塌陷 0.29-0.45× 在它眼里是 0.79-0.83×，过不了 0.35 阈）。
BLUR_ZONE_RATIO: float = 0.55      # 锐度 < 基准×此比例 → 记为塌陷帧
BLUR_COLLAPSE_RATIO: float = 0.35  # 塌陷最深要低于基准×此比例才算真模糊（防误伤天生偏软的段）
BLUR_RECOVER_RATIO: float = 0.5    # 窗内必须看到恢复到基准×此比例，才敢裁（看不到恢复宁少勿多）
# —— v0.4.1 色档收敛信号：注噪/taper 路的「收敛尾巴」是低频现象（重影+色档漂移），
# 锐度法不可见（L1 taper 实测：可见头部 3 帧亮度 0.270→0.282 爬升 + 首帧重影——GG 目检确认）。
# 旧像素注噪路裁 26=22+4 裁的就是它——本信号是它的观测端版本。
GRADE_DEV_Z: float = 4.0        # 体区 MAD 的 z 门槛
GRADE_DEV_FLOOR: float = 0.003  # 绝对下限（0-1 量纲；255 量纲测试由 z 项主导）
# —— v0.3.3 稳健统计层：参考分布来自段体（窗外体区），z 分数定位，深度比值做证据闸 ——
# 动机（GG 2026-09-13）：分辨率/步数/LoRA/场景内容都会整体移动锐度与帧差的绝对量级，
# 任何"绝对常数"都会在某个参数组合下失效。因此：
#   · 参考分布 = 段体自身（窗外 ≥8 帧，median + MAD）——4 步软渲染、运动模糊、平坦场景
#     自动进基准，零配置、零量纲；
#   · z 分数（|x-med|/(1.4826·MAD)）负责"是不是异常"，比值门槛（0.35/0.55/0.5，本来就
#     无量纲）负责"是不是深到值得动刀"——两道闸缺一不可：只有 z 会把"段体天生比头锐"
#     的正常渐变误裁，只有比值会在分布漂移时定位失准；
#   · 硬跳必须**验身**：跳后 2-5 帧锐度回到体分布才采信，否则视为假跳/闪烁交给锐度路。
Z_JUMP: float = 8.0                # 硬跳的稳健 z 门槛（对段体帧差 MAD 标准化；实测真跳 z≈90）
_LAP_KERNELS: dict = {}            # 按 device 字符串惰性缓存的 3×3 Laplacian 卷积核


def _robust_stats(x: torch.Tensor) -> Tuple[float, float]:
    """稳健 (median, 1.4826×MAD)。空输入返回 (0, 0)。"""
    if x.numel() == 0:
        return 0.0, 0.0
    med = float(x.median())
    mad = float((x - med).abs().median()) * 1.4826
    return med, mad


def _abs_max(t: torch.Tensor) -> float:
    """max(|t|)，不物化 abs() 的全量拷贝（float 张量上 max(|min|, max) 与之等价）。"""
    return max(-float(t.min()), float(t.max()))


def _baseline_floor(images: torch.Tensor) -> float:
    """帧差基线的保护下限，随数据实际量纲缩放（0-1 产线 / 0-255 测试同一套阈值）。"""
    try:
        scale = _abs_max(images.detach())
    except Exception:
        return 0.0
    return BASELINE_FLOOR_REL * scale if scale > 0 else 0.0


def _sharpness(images: torch.Tensor) -> torch.Tensor:
    """逐帧高频能量代理：3×3 **Laplacian 响应的均方**（≈锐度方差）。纯 torch，不依赖 cv2/PIL。

    带纹理的画面显著大于 0；重绘发虚（模糊）先杀高对比边缘，平方度量把它放大
    （实测：HD mild 塌陷帧在方差度量下 0.29-0.45×基准，mean-abs 度量下只见 0.79-0.83×）。
    """
    f = images.to(torch.float32)
    if f.dim() != 4:
        return torch.zeros(0)
    g = f.mean(dim=-1).unsqueeze(1)                    # [N,1,H,W]
    # 按 device 分键缓存：多 GPU / 多设备交替时不会来回重建，也无全局竞态
    key = str(g.device)
    kern = _LAP_KERNELS.get(key)
    if kern is None or kern.device != g.device:
        kern = g.new_tensor([[0.0, 1.0, 0.0],
                             [1.0, -4.0, 1.0],
                             [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        _LAP_KERNELS[key] = kern
    # replicate pad（不能用 conv2d 自带的 zero pad：常数帧边界会吃出假响应，
    # 小分辨率合成测试里边界占比过半，整个度量直接反转）
    gp = torch.nn.functional.pad(g, (1, 1, 1, 1), mode="replicate")
    resp = torch.nn.functional.conv2d(gp, kern)
    return resp.pow(2).mean(dim=(1, 2, 3))             # [N]


def _frame_diffs(images: torch.Tensor) -> torch.Tensor:
    """相邻帧差的逐帧标量（纯 CPU、纯张量，不依赖 cv2/PIL）。"""
    f = images.to(torch.float32)
    if f.dim() == 4 and f.shape[-1] in (1, 3, 4):   # [N,H,W,C] → 按通道均值
        return (f[1:] - f[:-1]).abs().mean(dim=(1, 2, 3))
    return (f[1:] - f[:-1]).abs().flatten(1).mean(dim=1)


def scan_head_jump(images: torch.Tensor, scan: int = 40) -> Tuple[int, float, float]:
    """段首扫描的**原始观测**：返回 ``(argmax 下标, 该处帧差, 段内基线)``。

    不做显著性判断、不加范围限制 —— 由上层按各自口径解释：
    ``find_head_jump`` 只判显著性，``describe_head_jump`` 再叠加"可执行范围"。
    """
    n = int(images.shape[0])
    if n < 4:
        return -1, 0.0, 0.0
    # ★ 先切窗再做差：整段物化两遍全量帧差，在 120+ 帧 / 768×448 上是 ~1GB 的瞬时峰值
    hi = min(scan, n - 1)
    if hi <= 2:
        return -1, 0.0, 0.0
    diff = _frame_diffs(images[:hi + 1])
    baseline = float(diff[2:hi].median())
    seg = diff[:hi]
    j = int(torch.argmax(seg).item())
    return j, float(seg[j].item()), baseline


def find_head_jump(images: torch.Tensor, scan: int = 40) -> Tuple[int, float, float]:
    """在前 ``scan`` 帧内找「本段起点」处的突变。

    返回 ``(jump_index, jump_mae, baseline_mae)``：
      - ``baseline_mae``：段内相邻帧差的中位数（跳过前 2 帧）
      - ``jump_mae``：``images[jump_index]`` 与后一帧的差
      - ``jump_index``：突变发生在前一帧的下标（即"应当再往前裁 1 帧"的位置）
    找不到突变时返回 ``(-1, 0.0, baseline)``。

    纯 CPU、纯张量：不依赖 cv2/PIL，可在节点里直接调。
    """
    j, jump, baseline = scan_head_jump(images, scan)
    if j < 0:
        return -1, jump, baseline
    if jump > JUMP_RATIO * max(baseline, _baseline_floor(images)):
        return j, jump, baseline
    return -1, jump, baseline


def detect_settle(
    images: torch.Tensor,
    pin: int,
    max_settle: int = MAX_SETTLE,
    ratio: float = JUMP_RATIO,
) -> Tuple[int, float, float]:
    """在**未裁剪**的段首附近量出真实切换点，返回建议的沉降帧数。

    ``images`` 是完整 decode 结果（长度 N，前 ``pin`` 帧是钉住区的复现）。
    返回 ``(settle, signal, reference)``——signal/reference 的量纲随判定路径不同：
    硬跳路 = 帧差，锐度路 = Laplacian 均方，色档路 = RGB 均值偏离；全**纯比值判定**。
    三路独立出候选，**取 settle 最大者**（不同伪影类型互补，谁检测到得深听谁的）。

    v0.4.1 四层判定（GG 要求的多维度/自参考方案）：

      0. **结构性护栏**（不变）：窄窗贴 ``pin`` 不做全局 argmax；``settle ≤ max_settle``；
         测不准 ⇒ 0 ⇒ 与"只裁钉住区"的旧行为逐位一致，最坏不会更差。
      1. **硬跳路（帧差 · 双门槛 + 验身）**：窗内最大帧差须同时过
         「体 z 分数 > Z_JUMP」（对段体帧差的 median+MAD 标准化——运动剧烈的段
         体差大，同样的跳不值钱）和「比值 > JUMP_RATIO×体中位」两道门；
         再**验身**：跳后 2-5 帧锐度须回到体分布（无一帧深塌陷）——跳完还是糊的
         = 假跳/闪烁，否决。
      2. **锐度路（塌陷-恢复 · 深度证据闸）**：基准 = max(复现区中位, 窗外体区中位)
         （v0.3.2 双基准：复现区自身可能已发软）；窗内须**同时**出现
         「塌陷帧（<0.55×基准）」和「深塌陷证据（最深处 <0.35×基准）」，
         且窗内可见恢复（>0.5×基准，宁少勿多）——沉降 = 最后一个塌陷帧 − pin + 1。
      3. **色档收敛路（v0.4.1，新增）**：注噪/taper 续接的「收敛尾巴」——可见头部
         若干帧的亮度/色档仍在对齐去噪轨迹（重影+低频漂移），锐度法不可见。
         基准 = 窗外体区 RGB 均值的**逐通道中位数**；阈值 = max(GRADE_DEV_Z×体MAD,
         绝对下限)。**必须在窗内观察到收敛**（头部偏离 → 其后全部回归基准）才动刀：
         没有收敛 = 头部本来就是另一档内容的正常延续（如窗外远处切镜），照裁会
         把合法内容裁掉（12.9 类反例）。沉降 = 首个回归帧的下标。
      4. **参考分布全部来自段体**：分辨率/步数/LoRA/内容整体移动量级时，基准同步
         移动——4 步低清、928p 高清、CombatV2、平坦场景共用同一套比值。
    """
    pin, max_settle = int(pin), max(0, int(max_settle))
    if pin <= 0 or max_settle <= 0:
        return 0, 0.0, 0.0
    n = int(images.shape[0])
    if n < pin + 2:
        return 0, 0.0, 0.0
    # ★ 只算用得到的那一小段，别对整段做差（120 帧 448x768 实测 39 ms vs 187 ms）
    win = images[: min(n, pin + max_settle + 2)]
    far_lo = pin + max_settle + 2
    body = images[far_lo: min(n, far_lo + 40)]
    if body.shape[0] < 8:
        body = win                      # 短段：体参考退化到窗内（仍优于任何全局常数）
    diff = _frame_diffs(win)
    inner = diff[2:max(3, pin - 1)]
    baseline = float(inner.median()) if inner.numel() else float(diff.median())
    lo = max(2, pin - 1)
    hi = min(int(diff.shape[0]), pin + max_settle)
    if hi <= lo:
        return 0, 0.0, baseline
    seg = diff[lo:hi]
    j = lo + int(torch.argmax(seg).item())
    val = float(diff[j].item())

    best = (0, 0.0, baseline)           # (settle, signal, reference)——取最大

    # —— 路径 1：硬跳（体 z 分数 + 比值双门槛，过了再验身）——
    b_med, b_mad = _robust_stats(_frame_diffs(body))
    z_denom = max(b_mad, _baseline_floor(win))
    z_jump = (val - b_med) / z_denom if z_denom > 0 else 0.0
    if z_jump > Z_JUMP and val > ratio * max(b_med, _baseline_floor(win)):
        sharp_all = _sharpness(images[: min(n, pin + max_settle + 8)])
        ref_all = max(float(sharp_all[:pin].median()),
                      _robust_stats(_sharpness(body))[0])
        accept = ref_all <= 0.0         # 锐度基准退化 → 无法验身 → 采信跳点
        if not accept:
            after_jump = sharp_all[j + 1: min(j + 5, int(sharp_all.shape[0]))]
            accept = after_jump.numel() == 0 or not bool(
                (after_jump < BLUR_COLLAPSE_RATIO * ref_all).any())
        if accept:
            s = max(0, min(max_settle, j + 1 - pin))
            if s > best[0]:
                best = (s, val, baseline)

    # —— 路径 2：锐度塌陷-恢复（双基准 + 深度证据闸）——
    sh = _sharpness(win)
    if sh.numel() >= pin + 2:
        far = _sharpness(images[far_lo: min(int(images.shape[0]), far_lo + 40)])
        ref = max(float(sh[:pin].median()),
                  float(far.median()) if far.numel() else 0.0)
        if ref > 0.0:
            zone = sh[pin: pin + max_settle + 1]
            below = zone < BLUR_ZONE_RATIO * ref
            dip = float(zone.min())
            if bool(below.any()) and dip < BLUR_COLLAPSE_RATIO * ref:
                last = int(below.nonzero()[-1].item())
                after = zone[last + 1:]
                if after.numel() > 0 and float(after.max()) >= BLUR_RECOVER_RATIO * ref:
                    s = min(max_settle, last + 1)
                    if s > best[0]:
                        best = (s, dip, ref)

    # —— 路径 3：色档收敛（v0.4.1）——头部偏离体区色档、且窗内可见回归才动刀——
    rgb_head = win.float().mean(dim=(1, 2))                       # [Nw,3]（量纲随输入）
    rgb_body = body.float().mean(dim=(1, 2))                      # [Nb,3]
    if rgb_head.shape[0] >= pin + 1 and rgb_body.shape[0] >= 4:
        ref_rgb = rgb_body.median(dim=0).values
        devs_head = (rgb_head[pin: pin + max_settle + 1] - ref_rgb).abs().mean(-1)
        devs_body = (rgb_body - ref_rgb).abs().mean(-1)
        _, b_mad_g = _robust_stats(devs_body)
        # 量纲下限用已切出的小窗 + 体区估计，不对整段物化 abs()
        scale = max(_abs_max(win), _abs_max(body)) or 1.0
        thr = max(GRADE_DEV_Z * b_mad_g, GRADE_DEV_FLOOR * scale)
        ok = devs_head <= thr
        if bool(ok.any()):
            c = int(ok.nonzero()[0].item())            # 首个回归基准的帧（窗内偏移）
            if bool(ok[c:].all()):                     # 其后全部回归 = 观察到收敛
                s = min(max_settle, c)
                if s > best[0]:
                    best = (s, float(devs_head[c - 1]) if c > 0 else 0.0, float(thr))

    return best


def describe_head_jump(images: torch.Tensor, scan: int = 40,
                       within: Optional[int] = None) -> str:
    """给日志用的一行接缝自检结论。

    ``within`` 之内才给"再裁 N 帧"这种**可执行**的建议（默认 ``ADVISE_WITHIN``）。
    超出的突变只报位置与数值 —— 那多半是本段自己的真实切镜而不是接缝，
    照它动刀会把新内容裁掉（v0.3.0 之前就是这个行为）。
    """
    if within is None:
        within = ADVISE_WITHIN
    j, jump, baseline = scan_head_jump(images, scan)
    floor = max(baseline, _baseline_floor(images))
    ratio = jump / floor if floor > 0 else 0.0
    if j < 0 or jump <= JUMP_RATIO * floor:
        return ("[H3 Relay] 接缝自检：前 %d 帧无突变（最大帧差 %.2f，段内基线 %.2f）→ 起点干净。"
                % (scan, jump, baseline))
    if j > int(within):
        return ("[H3 Relay] ⚠ 接缝自检：第 %d→%d 帧有突变（%.2f vs 基线 %.2f，比值 %.1f×），"
                "但位置超出自动沉降上限（%d 帧）→ 大概率是本段自己的切镜，不是接缝，不动刀。"
                % (j, j + 1, jump, baseline, ratio, MAX_SETTLE))
    return ("[H3 Relay] ⚠ 接缝自检：第 %d→%d 帧有突变（%.2f vs 基线 %.2f，比值 %.1f×）\n"
            "            → 把「续接裁重叠」的 settle_frames 从 -1（自动）改成 %d 再跑"
            "（多裁掉突变前那帧）。\n"
            "            别改 trim_frames —— 那一格已被连线接管，前端会藏起来，改不了。"
            % (j, j + 1, jump, baseline, ratio, j + 1))


def describe_latent(latent: Any) -> str:
    """一行摘要，用于日志与节点输出。"""
    try:
        parts = streams_from_latent(latent)
    except Exception as e:
        return "无法解析 latent：%s" % e
    bits = []
    for i, t in enumerate(parts):
        name = "视频" if i == 0 else ("音频" if i == 1 else "#%d" % i)
        bits.append("%s%s" % (name, tuple(t.shape)))
    v = parts[0]
    if v.ndim >= 5:
        bits.append("%d 帧 @ %dx%d" % (pixel_frames(int(v.shape[2])), int(v.shape[4]) * 16, int(v.shape[3]) * 16))
    return " | ".join(bits)


# ---------------------------------------------------------------- 0.4.0 拷贝桥
# 机制出处（拆解吸收，均开放许可）：
#   · AIMixer/ComfyUI_MiniMaxH3_Director（Apache-2.0）——latent 硬拷贝 + 噪声掩码 +
#     音频尾拷贝 + NestedTensor 双流打包；
#   · comfyui-minimax-h3-audio-T8（native_masked_context）——掩码走 **ComfyUI 原生
#     H3 契约**（mask=0 保留 / 1 生成，MiniMaxH3.scale_latent_inpaint /
#     mask_row_values，0.30+），与具体采样器无关 → 通用兼容；
#     且 T8 用全 0 硬锁（钉住区零重绘）——与我们 0.3.x 实测「复现=漂移源」同向，
#     设为默认；Director 的 1.0→seam_min 锥形作为实验档保留。
#   · 消费端实测：SelfLiftH3Sampler 原生读 latent["noise_mask"]（[B,1,T,H,W]，
#     post-CFG hook 把 0 区每步钉回 clean anchor）——无需改任何采样器。
# 音频：尾随拷贝只作采样上下文（掩码只做视频流）；可见的声画拼接仍归 TrimAV/组装层。
SEAM_TAPER_TOKENS: int = 4
SEAM_MIN_MASK: float = 0.10
SEAM_MIN_FLOOR: float = 0.0
AUDIO_RELEASE_TICKS: int = 6
MASK_MODES: Tuple[str, ...] = ("hard", "taper")


def prefix_taper_weights(
    steps: int,
    taper: int = SEAM_TAPER_TOKENS,
    seam_min: float = SEAM_MIN_MASK,
) -> Tuple[float, ...]:
    """拷贝前缀的掩码权重：头部 1.0（**完全重绘**，反正被裁），线性降到缝端 seam_min。

    ⚠️ **这不是「软一点的硬锁」——taper 档下钉住区没有被钉住。**
    掩码语义是 ``denoised = 模型生成 * m + 上段尾 * (1-m)``：m=0 才钉住，m=1 是重绘。
    所以 taper 档每一帧都留 ``seam_min``~100% 的重绘自由度（seam_min=0.3 即缝端仍 30% 重绘），
    段首会被模型改写 ⇒ 观感「续不上」，而 TrimAV 仍按窗口帧数照裁 ⇒ 顺带裁掉真实剧情。
    本档**只用于「渐进接管」对照实验**；真续接请用 ``mask_mode="hard"``。
    （2026-09-15：产线曾误把 taper 当默认跑了 23 次，见 CHANGES 0.4.2 文档节。）
    """
    n = int(steps)
    if n < 1:
        return ()
    taper = max(1, min(int(taper), n))
    head = n - taper
    floor = max(SEAM_MIN_FLOOR, min(1.0, float(seam_min)))
    weights = [1.0] * head
    weights.extend(1.0 + (floor - 1.0) * (float(i + 1) / float(taper)) for i in range(taper))
    return tuple(weights)


def _nested_pair(video: torch.Tensor, audio: torch.Tensor, template: Any = None) -> Any:
    """打包 AV 双流为 NestedTensor（ComfyUI 运行时），离线环境退化为 list。"""
    try:
        from comfy.nested_tensor import NestedTensor

        return NestedTensor((video, audio))
    except Exception:
        cls = type(template) if template is not None else None
        if cls is not None and cls is not torch.Tensor:
            try:
                return cls((video, audio))
            except Exception:
                pass
    return [video, audio]


def build_continue_latent(
    target: Any,
    prev: Any,
    frames: int,
    mask_mode: str = "hard",
    taper: int = SEAM_TAPER_TOKENS,
    seam_min: float = SEAM_MIN_MASK,
    pin_audio: bool = True,
) -> Tuple[Dict[str, Any], int, str]:
    """0.4.0 拷贝桥：把上一段 AV 尾部**逐位拷贝**进本段初始 latent + 噪声掩码。

    返回 ``(latent, covered, report)``；``covered`` = 应裁帧数（接 TrimAV 的 trim_frames）。

    与 conditioning 钉帧（H3RelayMotionContext）的本质区别：钉住区**不重绘**——
    mask=0 区每步被采样器钉回拷贝进来的上段尾部 latent，0.3.x 实测的
    「复现发糊/漂移」这一类伪影从机制上消失。三条硬约束（违反即 raise）：
      1. ``frames`` 落在 5+17k 网格且尾段起点 5-token 对齐（复用 video_tail_from_latent）；
      2. 上段与本段分辨率一致；
      3. 拷贝前缀必须给新内容留至少 1 个 token。
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(
            "mask_mode 只认 %s，得到 %r" % (" / ".join(MASK_MODES), mask_mode)
        )
    tv = video_from_latent(target)
    pv = video_from_latent(prev)
    if (int(tv.shape[3]), int(tv.shape[4])) != (int(pv.shape[3]), int(pv.shape[4])):
        raise ValueError(
            "拷贝桥禁止跨分辨率：上段 %dx%d ≠ 本段 %dx%d"
            % (int(pv.shape[4]) * 16, int(pv.shape[3]) * 16,
               int(tv.shape[4]) * 16, int(tv.shape[3]) * 16)
        )
    blocks, _offsets, covered = video_tail_from_latent(prev, frames)
    steps = len(blocks)
    total_t = int(tv.shape[2])
    if steps >= total_t:
        raise ValueError(
            "拷贝前缀 %d 步占满本段 %d 步 → 没有新内容可生成；缩短 context_frames 或加长本段"
            % (steps, total_t)
        )

    # tv.clone() 是必须的：下面要原位写前缀，不能污染调用方的 latent。
    # 峰值 ≈ 1×target 视频流（latent 在 GPU 时即 1× 显存，尾段 blocks 本身已占 steps/total）。
    video = tv.clone()
    tail_v = torch.cat(blocks, dim=2).to(device=video.device, dtype=video.dtype)
    video[:, :, :steps] = tail_v

    audio = None
    rt = 0
    if pin_audio:
        # 只有真的要钉音频时才要求 target 带音频流——pin_audio=False
        # 必须允许纯视频 latent 走通（参数语义）。
        audio = audio_from_latent(target).clone()
        a_tail, rt, _overhang, _raw, _grid_off = audio_tail_from_latent(
            prev, int(frames), pixel_frames(int(pv.shape[2])))
        rt = max(0, min(int(rt), int(audio.shape[-1]) - 1))
        if rt > 0:
            audio[..., :rt] = a_tail[..., :rt].to(device=audio.device, dtype=audio.dtype)
    else:
        try:
            audio = audio_from_latent(target)     # 有则原样保留（不拷贝、不改动）
        except ValueError:
            audio = None                          # 纯视频桥：只出视频流

    # 掩码与 latent 同设备同 batch：GPU latent + CPU 掩码会在采样器里炸或静默错位
    vmask = torch.ones((int(tv.shape[0]), 1, total_t, int(tv.shape[3]), int(tv.shape[4])),
                       dtype=torch.float32, device=tv.device)
    if mask_mode == "hard":
        vmask[:, :, :steps] = 0.0
        mask_desc = "硬锁（全 0，钉住区零重绘）"
    else:
        w = prefix_taper_weights(steps, taper, seam_min)
        # 权重直接建在掩码设备上，省一次跨设备拷贝（latent 在 GPU 时 vmask 也在 GPU）
        vmask[:, :, :steps] = torch.tensor(w, dtype=torch.float32,
                                           device=vmask.device).view(1, 1, steps, 1, 1)
        mask_desc = "锥形 %.2f→%.2f（taper=%d）" % (w[0], w[-1], taper)

    out = dict(target) if isinstance(target, dict) else {}
    if audio is None:
        out["samples"] = video          # 纯视频路径：不打包 NestedTensor
    else:
        out["samples"] = _nested_pair(video, audio, tv)
    out["noise_mask"] = vmask
    report = (
        "[H3 Relay] 拷贝桥：写入 %d 步（%d 帧）视频尾 + %d 音频 tick（上下文用%s）；"
        "掩码 %s；trim=%d"
        % (steps, covered, rt,
           "" if audio is not None else "／本段无音频流", mask_desc, covered)
    )
    return out, int(covered), report
