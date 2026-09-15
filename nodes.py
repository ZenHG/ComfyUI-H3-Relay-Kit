# -*- coding: utf-8 -*-
"""H3 Relay Kit · 节点层

六个节点，覆盖"用作者的续接方式"所需的全部接线：

  🔗 H3 续接 Latent 存   —— 把本段的 AV latent 落盘，供下一段读
  🔗 H3 续接 Latent 读   —— 读回上一段的 AV latent
  🔗 H3 续接 Latent 桥   —— 把上一段尾段钉进本段 conditioning（latent 直取，零重编码）
  🔗 H3 续接 拷贝桥      —— 上一段尾部**逐位拷贝**进本段初始 latent + 噪声掩码（钉住区不重绘）
  🔗 H3 续接裁重叠        —— 裁掉续接段头部的重叠帧（视频 + 音频同裁）
  🔗 H3 续接连跑 Chain    —— 同分组框内自动推进「桥 + 落盘」段号并排队连跑

两条续接路线**二选一**，不可同图串联：
  · Latent 桥（conditioning 钉帧）→ 模型重绘上一段（有复现漂移风险，检测兜底）；
  · 拷贝桥（latent + 噪声掩码）→ 钉住区零重绘（0.4.0 起，通用兼容不绑采样器）。

接线（替换像素续接时）：
    CSGlideCastCS[0] ─ conditioning ─┐
    CSGlideCastCS[1] ─ latent ───────┤
    H3 续接 Latent 读 ─ context ─────┤→ 🔗 续接 Latent 桥 [0] → 采样器 positive
    （上一段：采样器 latent → 🔗 续接 Latent 存）

注意：走任一桥时，上游的**像素续接字段（如 H3 Studio 的 cont）必须留空**，
否则两套续接都会往 minimax_keyframes 里塞锚，画面会打架。
"""

from __future__ import annotations

import math
import os

import folder_paths

from . import relay_core as CORE
from . import layout_contract as CONTRACT


CATEGORY = "H3 Relay Kit"

# 落盘根目录：ComfyUI/output/relay_kit/
_RELAY_ROOT = os.path.join(folder_paths.get_output_directory(), "relay_kit")

# Windows 保留设备名：作为目录名会让 makedirs 抛裸 OSError
_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {"COM%d" % i for i in range(1, 10)}
    | {"LPT%d" % i for i in range(1, 10)}
)


def _stage_path(run_id: str, stage_index: int) -> str:
    """按 run 标识 + 段号推导落盘路径。段号从 0 开始。"""
    # 非法字符替换为 _（而不是删除）—— 否则 "my/film" 与 "myfilm" 会静默撞进同一目录
    rid = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(run_id))
    if not rid.strip("_"):
        raise RuntimeError("run_id 不能为空（用来把同一部片子的各段归到一个目录）。")
    if rid.upper() in _WIN_RESERVED:
        rid += "_"
    idx = int(stage_index)
    if idx < 0:
        raise RuntimeError("stage_index 不能为负（段号从 0 开始，第 1 段=0）。")
    return os.path.join(_RELAY_ROOT, rid, "stage_%05d.safetensors" % idx)


class H3RelayLatentSave:
    """把本段采样器输出的 AV latent 落盘。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "【接法】从本段采样器（SelfLiftH3Sampler）的 latent 输出口拉线过来。\n"
                               "作用：把本段拍完的 latent 存成文件，它是下一段的「接力棒」。",
                }),
                "run_id": ("STRING", {
                    "default": "relay",
                    "tooltip": "【填什么】这部片子的名字，比如 myfilm、ep01。\n"
                               "⚠ 必须和「续接 Latent 桥」上的 run_id 一字不差，否则桥找不到文件。\n"
                               "换新片子必须换新名字：同一个名字重跑同段号会覆盖旧文件！\n"
                               "文件存到：ComfyUI/output/relay_kit/<run_id>/",
                }),
                "stage_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "【填什么】本段是全片的第几段。第 1 段填 0，第 2 段填 1，第 3 段填 2…\n"
                               "⚠ 必须和「续接 Latent 桥」上的 stage_index 一样大。\n"
                               "改段号时两个节点都要改（用 Chain 节点可以自动改）。",
                }),
                "note": ("STRING", {
                    "default": "",
                    "tooltip": "【可留空】随手写个备注（比如「22帧窗 v2」），存进文件里方便事后分辨版本。",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "path")
    FUNCTION = "save"
    CATEGORY = CATEGORY
    # 没有下游消费时也必须执行 —— 落盘本身就是它的产物。
    OUTPUT_NODE = True
    DESCRIPTION = "把本段 AV latent 落盘，供下一段做 latent 续接（零重编码）。"

    def save(self, latent, run_id, stage_index, note=""):
        path = _stage_path(run_id, stage_index)
        CORE.save_av_latent(latent, path, note=note)
        size_mb = os.path.getsize(path) / 1024 ** 2
        print(
            "[H3 Relay] 已存 stage %d → %s (%.1f MB)\n            %s"
            % (int(stage_index), path, size_mb, CORE.describe_latent(latent))
        )
        return (latent, path)


class H3RelayLatentLoad:
    """读回上一段的 AV latent，供续接。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_id": ("STRING", {
                    "default": "relay",
                    "tooltip": "【填什么】与「续接 Latent 存」一致的片子名。\n"
                               "⚠ 不一致 = 找不到上一段的文件，直接报错。",
                }),
                "stage_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "【填什么】填「你要读的那一段」的段号 = 本段段号 - 1。\n"
                               "例：现在做第 2 段（本段号 1），这里填 1 → 读第 1 段（stage 0）的文件。\n"
                               "填 0 会报错：第 1 段没有上一段可读。",
                }),
            },
            "optional": {
                "explicit_path": ("STRING", {
                    "default": "",
                    "tooltip": "留空则按 run_id + 段号自动推导；填了就直接读这个文件（用于断点续跑换源）。",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("context_latent", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "读回上一段的 AV latent；第 1 段（stage_index=0）没有上一段时会明确报错。"

    def load(self, run_id, stage_index, explicit_path=""):
        idx = int(stage_index) - 1
        explicit = (explicit_path or "").strip()
        # 必须先判段号再进 _stage_path：否则 stage_index=0 会先撞上
        # 「stage_index 不能为负」的误导性报错，下面的引导文案永远到不了。
        if idx < 0 and not explicit:
            raise RuntimeError(
                "stage_index=0 是第 1 段，没有上一段可续。\n"
                "    第 1 段请走独立路径（不接本节点，或把续接 Latent 桥的 context_latent 留空）。"
            )
        path = explicit or _stage_path(run_id, idx)
        latent = CORE.load_av_latent(path)
        info = CORE.describe_latent(latent)
        print("[H3 Relay] 已读 stage %d ← %s\n            %s" % (idx, path, info))
        return (latent, info)


class H3RelayMotionContext:
    """latent 桥：把上一段尾段钉进本段 conditioning。

    与像素续接（mp4 → VAE 重编码）走同一个原生协议（minimax_keyframes），
    区别是本节点**不重编码** —— 直接用上一段采样时的 latent。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "【接法】从本段的出词/参考条件节点（如官方 MiniMaxH3ReferenceToVideo 的 positive 口）拉线过来。\n"
                               "作用：告诉模型本段要拍什么。桥会在它里面悄悄塞进「上一段的结尾」。",
                }),
                "latent": ("LATENT", {
                    "tooltip": "【接法】同一个条件节点的 latent 输出口拉线过来。\n"
                               "作用：提供本段的规格（分辨率/帧数）。⚠ 必须和上一段分辨率一样，不一样直接报错。",
                }),
                "trim_frames": ("INT", {
                    "default": 22, "min": 5, "max": 124, "step": 17,
                    "tooltip": "【填什么】让模型看着上一段结尾多少帧来接戏。\n"
                               "  · 22（默认）= 标准，0.9 秒衔接上下文，最稳\n"
                               "  · 5 = 最小，成片新内容多 1 秒，但静态场景衔接变弱\n"
                               "  · 只能填 5/22/39/56/73/90/107/124，别的数直接报错\n"
                               "【不用管的部分】钉住的帧成片里会被自动裁掉（第 3 路输出同步给裁节点）。",
                }),
            },
            "optional": {
                "context_latent": ("LATENT", {
                    "tooltip": "【不用接线，留空！】只要 run_id 填了、stage_index ≥ 1，\n"
                               "桥就会自己去 output/relay_kit/<run_id>/ 读上一段的 latent 文件。\n"
                               "这个口是给高级用法手动连「续接 Latent 读」节点用的。",
                }),
                "audio_frames": ("INT", {
                    "default": 0, "min": 0, "max": 362, "step": 1,
                    "tooltip": "【不用动，保持 0】音频跟着视频窗走（钉住同样长的声音，让音乐/环境声接着往下走而不是重新起头）。\n"
                               "想单独加长音频上下文才填别的数（0 = 跟随视频窗）。",
                }),
                "run_id": ("STRING", {
                    "default": "relay",
                    "tooltip": "【填什么】这部片子的名字（和「续接 Latent 存」上一致）。\n"
                               "  · 第 2 段起，桥自动去 ComfyUI/output/relay_kit/<run_id>/ 读上一段的 latent 文件\n"
                               "  · ⚠ 两个节点上名字不一致 = 找不到文件\n"
                               "  · 换新片子必须换名字，重跑会覆盖同段号旧文件",
                }),
                "stage_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "【填什么】本段是第几段。\n"
                               "  · 第 1 段 → 填 0（直通：不续接，只把 latent 交给落盘节点存档）\n"
                               "  · 第 2 段 → 填 1（自动读第 1 段的文件来续接）\n"
                               "  · 第 3 段 → 填 2……以此类推\n"
                               "⚠ 和「续接 Latent 存」上的数保持一样大；用 Chain 节点可自动推进，不用手改。",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "INT")
    RETURN_NAMES = ("conditioning", "report", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "latent 桥续接（零重编码）：上一段尾段按原样钉进本段，画面声音都接着走。\n"
        "手把手：第 1 段 stage_index=0 跑一遍 → 第 2 段把桥和落盘的 stage_index 都改成 1 → 换 prompt 再跑。"
    )

    def apply(self, conditioning, latent, trim_frames=22, context_latent=None,
              audio_frames=0, run_id="", stage_index=0):
        bad = CORE.self_check()
        if bad:
            raise RuntimeError("H3 Relay 网格自检失败：\n    " + "\n    ".join(bad))
        CONTRACT.enforce()   # 上游 ComfyUI 改了 H3 网格 → 在这里拒绝，而不是产出坏片子

        idx = int(stage_index)
        auto_note = ""
        if context_latent is None and (run_id or "").strip():
            if idx >= 1:
                path = _stage_path(run_id, idx - 1)
                context_latent = CORE.load_av_latent(path)
                auto_note = "自动读上一段 ← %s" % path
                print("[H3 Relay] " + auto_note)
            else:
                auto_note = "stage_index=0（第 1 段）→ 直通不续接。"

        # 说清了是第 N 段（N≥2）却拿不到上一段 —— 绝不能悄悄降级成"独立段"：
        # 那样工作流会一路绿灯跑完，产出的却是没有续接的哑剧式接缝。
        # 这是本包唯一一处"用户忘填"会导致静默坏片的路径，故硬拦。
        if context_latent is None and idx >= 1:
            raise RuntimeError(
                "stage_index=%d 表示本段是第 %d 段，但没有可续接的上一段 latent：\n"
                "    · context_latent 没接线，且\n"
                "    · run_id 是空的（或只有空白）\n"
                "再跑下去会「静默直通」—— 产出的是独立段而不是续接段，"
                "但界面与日志都显示成功。\n"
                "    第 2 段起请把 run_id 填成与「续接 Latent 存」完全一致的名字；\n"
                "    若这确实是独立段，把 stage_index 改回 0。"
                % (idx, idx + 1)
            )

        if context_latent is None:
            msg = "[H3 Relay] 无 context_latent → 直通（独立段，不续接）。" + (
                (" " + auto_note) if auto_note else "")
            print(msg)
            return (conditioning, msg, 0)   # 第 3 路 = 裁帧数；直通不裁

        plan = CORE.plan_relay(
            latent,
            context_latent,
            trim_frames=int(trim_frames),
            audio_frames=int(audio_frames) or None,
        )
        out = CORE.apply_relay(conditioning, plan)

        lines = ["[H3 Relay] latent 桥续接：" + plan.summary()]
        if auto_note:
            lines.append("    " + auto_note)
        lines.append("    本段 " + CORE.describe_latent(latent))
        lines.append("    上段 " + CORE.describe_latent(context_latent))
        for n in plan.notes:
            lines.append("    注记：" + n)
        report = "\n".join(lines)
        print(report)
        return (out, report, int(plan.trim))


class H3RelayTrimAV:
    """裁掉续接段头部的重叠帧（视频 + 音频同裁，A/V 不失步）。

    为什么必须裁：钉住区的前 ``trim_frames`` 帧是模型对上一段尾部的**重生成**
    （实测与原帧逐帧 MAE ≈ 6/255），属于过渡产物。不裁就拼，接缝处会看到
    约 0.9 秒的重播。

    口径与作者一致：裁 **decode 之后的像素帧**（不是裁 latent）——
    latent 的帧跨度按 token 在序列里的相位（k%5）决定，砍头会让相位错位。

    为什么还要多裁几帧（沉降）：钉住区之后模型还会先**复现**上一段若干帧，
    然后才切到本段 prompt；切换点**逐段不同**，可能落在钉住区之外。
    `settle_frames=-1`（默认）时本节点直接在这段已 decode 的画面上量出切换点，
    把它一并裁掉 —— 用户不需要配置任何东西。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "【接法】从本段的画面解码节点（VAEDecode）拉线过来。\n"
                               "作用：收到的是「含重复开头」的完整画面，本节点把重复部分裁掉。",
                }),
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 362, "step": 1,
                    "tooltip": "【不用填！】从「续接 Latent 桥」的第 3 路输出（trim_frames）拉线过来，全自动：\n"
                               "  · 第 1 段桥直通 → 自动 0（不裁）\n"
                               "  · 第 2 段起 → 自动 = 钉住帧数（如 22）\n"
                               "自己手填反而容易和桥对不上。",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001,
                    "tooltip": "【不用动】H3 固定 24 帧/秒，音频按这个换算着一起裁。",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "【务必接线】从音频解码节点（VAEDecodeAudio）拉线过来。\n"
                               "画面和声音按同一帧数一起裁，保证音画不串位。不接的话声音会比画面长出一截。",
                }),
                # ⚠ 新 widget 必须追加在**最后一个**：widgets_values 按位置对应，
                #    插在中间会让旧工作流里它后面的取值整体错位（见 CHANGES 0.2.1）。
                "settle_frames": ("INT", {
                    "default": -1, "min": -1, "max": 34, "step": 1,
                    "tooltip": "【不用动，保持 -1】钉住区后面，模型还会先「复现」上一段几帧才切到本段画面。\n"
                               "这几帧不裁掉，拼接处会看到 1 帧硬跳、上一段的字幕/文字被带过来。\n"
                               "  · -1（默认）= 自动：按本段实际画面量出该多裁几帧（推荐，不用管）\n"
                               "  ·  0 = 关闭（与 0.2.x 旧版行为一致）\n"
                               "  ·  N = 固定多裁 N 帧（想各段等长时用：全片填同一个数）\n"
                               "自动模式量不出来时会退回成 0，不会比旧版更差。\n"
                               "日志里「裁首 X 帧 = 钉住 Y + 沉降 Z」就是它的结果。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "report")
    FUNCTION = "trim"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "裁掉续接段头部的重叠帧，并自动把钉住区之后那段「复现帧」一并裁掉"
        "（settle_frames=-1 自动，无需配置）；视频与音频同裁，避免重播与音画失步。"
    )

    def trim(self, images, trim_frames=0, fps=24.0, audio=None, settle_frames=-1):
        CONTRACT.enforce()   # 裁帧算术同样依赖上游网格，先过契约
        # 服务端防线：widget 的 min=1.0 只挡 UI，API 提交 fps=0/NaN 会一路除到底
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = float("nan")
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(
                "fps 必须是 (0, +∞) 内的有限正数，得到 %r。\n"
                "    H3 固定 24 帧/秒——把「裁重叠」的 fps 改回 24 即可（音频按它换算着一起裁）。"
                % (fps,)
            )
        pin = int(trim_frames)
        before = int(images.shape[0])
        if pin <= 0:
            msg = "[H3 Relay] 裁 0 帧 → 不裁（独立段或纯首段）。"
            print(msg)
            return (images, audio, msg)

        # 沉降帧：钉住区之后模型还会先复现上一段若干帧才切到本段 prompt。
        # 切换点是**逐段不同的量**，所以默认让它自己量（-1），而不是让用户猜。
        want = int(settle_frames)
        if want < 0:
            settle, jump, base = CORE.detect_settle(images, pin)
            why = ("自动检测：切换信号 %.1f / 基准 %.1f（帧差突变/锐度塌陷/色档收敛）" % (jump, base)) if settle \
                else ("自动检测：未检出切换点（帧差 %.1f / 基线 %.1f，无突变亦无锐度塌陷）" % (jump, base))
        else:
            settle, why = want, "手动指定"
        if pin + settle >= before:
            settle = max(0, before - 1 - pin)
            why += "（已夹到本段长度上限）"

        n = pin + settle
        out = CORE.trim_head_frames(images, n)
        audio_out = CORE.trim_audio_head(audio, n, float(fps)) if audio is not None else None
        after = int(out.shape[0])
        line = ("[H3 Relay] 裁首 %d 帧 = 钉住 %d + 沉降 %d ｜ %s\n"
                "           画面 %d → %d 帧（%.3fs → %.3fs）"
                % (n, pin, settle, why, before, after, before / float(fps), after / float(fps)))
        if audio is not None:
            line += "；音频 %d → %d 采样点" % (
                int(audio["waveform"].shape[-1]), int(audio_out["waveform"].shape[-1]))
        else:
            line += "；⚠ 未接 audio，画面裁了但音频没裁 → 可能音画不同步"
        print(line)
        # 接缝自检：裁后起点若仍有突变，说明沉降量不够
        check = CORE.describe_head_jump(out)
        print(check)
        return (out, audio_out, line + "\n" + check)


class H3RelayChain:
    """🔗 续接连跑（Chain）—— 纯控制节点，不在执行路径上。

    前端按钮（web/relay_kit_chain.js）会找到同一张图里的
    「续接 Latent 桥 + 续接 Latent 存」，自动推进它们的 stage_index 并排队：

        ▶ Run      按当前段号跑一次（不满意可重跑，覆盖同段号文件）
        ✔ Approve  段号 +1（桥和落盘同步改），排队跑下一段
        ⏩ 连跑     按 segments 自动循环：跑完一段 → 段号+1 → 再跑（0 = 无限）
        ⏹ Stop     当前采样跑完后停止推进
        ↺ Reset    段号归 0，从第 1 段重来

    使用前提：把 Chain、桥、落盘三个节点拉进**同一个分组框**（框选 → 右键 → 添加分组），
    否则按钮找不到要推进的节点。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("INT", {
                    "default": 5, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "【连跑几段】点「⏩ 连跑」时生效：5 = 连跑 5 段后自动停；0 = 不停，直到点 ⏹ Stop。\n"
                               "其他按钮不受它影响。",
                }),
            },
            "optional": {
                # 只用于**显示**：前端把状态写进这一格。
                # 必须是**最后一个** widget —— 这样旧工作流里少这一格时只走默认值，
                # 不会让它前面的取值错位（见 CHANGES 0.2.1 的槽位错位说明）。
                "status": ("STRING", {
                    "default": "",
                    "tooltip": "【不用填，自动显示】前端把连跑状态写在这里：\n"
                               "当前段号 / 已排队 / ⚠ 分组没放对 / ⚠ 排队失败…\n"
                               "点了按钮没反应时，先看这一格说了什么。",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "自动连跑控制器：配合桥 + 落盘使用。\n"
        "第一步：把 Chain、桥、落盘放进同一个分组框。\n"
        "第二步：桥和落盘 stage_index 填 0，点 ▶ Run 拍第 1 段。\n"
        "第三步：点 ⏩ 连跑（或每段点 ✔ Approve），段号自动推进，不用再手改。\n"
        "状态显示在 status 格子里（点了没反应就看它）。"
    )

    def noop(self, segments=5, **kwargs):
        # **kwargs 吞掉 status 这类只用于显示的输入
        return {}


class H3RelayCopyBridge:
    """0.4.0 拷贝桥：上一段尾部 AV latent **逐位拷贝**进本段初始 latent + 噪声掩码。

    与 H3RelayMotionContext（conditioning 钉帧）二选一，不可同图串联：
      · Latent 桥（钉帧）：模型重绘上一段尾段 → 有复现漂移/发糊风险（0.3.x 实测），
        观测端沉降检测兜底；
      · 拷贝桥（本节点）：``mask_mode="hard"`` 时钉住区不重绘（掩码 0 区每步被钉回
        拷贝 latent），复现伪影这一类从机制上消失；掩码消费走 ComfyUI 原生 H3 契约与
        SelfLift 的 noise_mask 支持——**不绑定任何特定采样器**。
        ⚠️ ``mask_mode="taper"`` **不钉住**（每帧留 seam_min~100% 重绘自由度），只作对照实验档。
    输出 INT = 应裁帧数（=拷贝跨度），接 H3RelayTrimAV 的 trim_frames；
    TrimAV 的 settle_frames 保持 -1，观测端继续守接管帧。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "本段初始 AV latent（CSGlideCastCS / EmptyH3LatentAV 的 latent 输出）。\n"
                               "上一段尾部会逐位写进它的开头，并附噪声掩码。",
                }),
                "context_latent": ("LATENT", {
                    "tooltip": "上一段的完整 AV latent（🔗 H3 续接 Latent 读）。\n"
                               "第 1 段（stage 0）不要接本节点——没有上一段可拷。",
                }),
                "context_frames": ("INT", {
                    "default": 22, "min": 5, "max": 124, "step": 17,
                    "tooltip": "拷贝窗口帧数，只认 5+17k 网格（5/22/39/56/73/90/107/124）。\n"
                               "须小于本段帧数（前缀必须给新内容留位置）。",
                }),
            },
            "optional": {
                "mask_mode": (["hard", "taper"], {
                    "default": "hard",
                    "tooltip": "掩码语义：denoised = 模型生成 * m + 上段尾 * (1-m)。**m=0 才钉住，m=1 是重绘。**\n"
                               "hard = 全窗 m=0（钉住区零重绘，默认，真续接用这个）；\n"
                               "taper = 头部 m=1.0（**完全重绘**）线性降到缝端 seam_min\n"
                               "        —— ⚠ **钉住区实际上没有被钉住**，只是给模型一个软提示；\n"
                               "        seam_min=0.3 意味着连缝端都留 30% 重绘。\n"
                               "        仅用于「渐进接管」对照实验；期望真续接请保持 hard。",
                }),
                "taper_tokens": ("INT", {
                    "default": 4, "min": 1, "max": 12, "step": 1,
                    "tooltip": "仅 taper 模式：缝端前多少个 token 参与线性过渡。",
                }),
                "seam_min": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "仅 taper 模式：缝端掩码下限（m 值）。\n"
                               "0 = 缝端完全硬锁；>0 表示缝端仍留同等比例的重绘自由度\n"
                               "（0.3 即缝端 30% 重绘）。注意它只管缝端——头部恒为 1.0 全重绘。",
                }),
                "pin_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "把上一段音频尾也拷进本段音频 latent 开头（采样上下文用）。\n"
                               "掩码只做视频流；可见的声画拼接仍归「裁重叠」与组装层。",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "INT")
    RETURN_NAMES = ("latent", "report", "trim_frames")
    FUNCTION = "bridge"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "把上一段尾部 AV latent 逐位拷进本段开头并附噪声掩码（钉住区不重绘），\n"
        "消除「复现发糊/漂移」这一类接缝伪影。输出 trim_frames 接「裁重叠」。\n"
        "⚠️ 与「Latent 桥」（conditioning 钉帧）二选一，不可同图串联。"
    )

    def bridge(self, latent, context_latent, context_frames,
               mask_mode="hard", taper_tokens=4, seam_min=0.10, pin_audio=True):
        CONTRACT.enforce()
        out, covered, report = CORE.build_continue_latent(
            latent, context_latent, int(context_frames),
            mask_mode=mask_mode, taper=int(taper_tokens),
            seam_min=float(seam_min), pin_audio=bool(pin_audio),
        )
        print(report, flush=True)
        return (out, report, covered)


NODE_CLASS_MAPPINGS = {
    "H3RelayLatentSave": H3RelayLatentSave,
    "H3RelayLatentLoad": H3RelayLatentLoad,
    "H3RelayMotionContext": H3RelayMotionContext,
    "H3RelayCopyBridge": H3RelayCopyBridge,
    "H3RelayTrimAV": H3RelayTrimAV,
    "H3RelayChain": H3RelayChain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RelayLatentSave": "🔗 H3 续接 Latent 存",
    "H3RelayLatentLoad": "🔗 H3 续接 Latent 读",
    "H3RelayMotionContext": "🔗 H3 续接 Latent 桥",
    "H3RelayCopyBridge": "🔗 H3 续接 拷贝桥",
    "H3RelayTrimAV": "🔗 H3 续接裁重叠",
    "H3RelayChain": "🔗 H3 续接连跑 Chain",
}
