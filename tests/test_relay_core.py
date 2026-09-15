# -*- coding: utf-8 -*-
"""H3 Relay Kit 离线单测（零 GPU、零模型、秒级）

跑法（在包目录下）：
    python tests/test_relay_core.py

脚本会自动往上找到 ComfyUI 根目录（本包装在 ``<ComfyUI>/custom_nodes/`` 下时）。
装在别处 / 只 clone 了本包时，用环境变量指定：

    COMFYUI_PATH=/path/to/ComfyUI python tests/test_relay_core.py

只需要 ``torch`` 与 ``safetensors``；不加载任何模型、不碰显存。
但需要能 import 到 ComfyUI（comfy.nested_tensor / node_helpers / folder_paths）。

覆盖：
  1. 时序网格自洽（22 帧=7 步 / 39 帧=12 步 / 192 帧=57 步）
  2. 尾段切片逐位正确 + 起始必须落在 5 步周期边界
  3. 硬错误：分辨率不一致 / 非网格帧数 / 窗口大于段长 → 必须 raise
  4. conditioning 注入：keyframes 合并、钉住区旧锚丢弃、音频 ref 追加
  5. AV latent 落盘往返一致
  6. 裁头重叠：画面音频同裁逐位正确、时长对齐、越界 raise
  7. 节点返回值契约：每个分支返回路数 == len(RETURN_TYPES)
  8. 接缝自检 find_head_jump / describe_head_jump
  9. 段号声明与取源矛盾必须 raise（不得静默直通）
 10. streams_from_latent 对非 NestedTensor 的健壮性（不得按 batch 维误拆）
 11. H3RelayChain 的 status 槽位与前端一致
 12. 沉降帧 settle：pin 与 crop 解耦 + 自动检测（观测端决定，用户零配置）
 13. 沉降三路检测：硬跳验身 / 锐度塌陷-恢复 / 双基准与量纲不变
 14. 拷贝桥（0.4.0）：位级拷贝 + 噪声掩码 + raise 防线
 15. 色档收敛信号（0.4.1）：注噪/taper 收敛尾巴的观测端
 16. 0.4.2 回归：导出音频分支 / 中文 note / 服务端校验 / 掩码设备 / 契约降级缓存
"""

import os
import sys
import tempfile
import traceback

import torch

# ---------------------------------------------------------------- 定位 ComfyUI
# 本包通常装在 <ComfyUI>/custom_nodes/<本包>/，往上两级就是 ComfyUI 根。
# 装在别处时用 COMFYUI_PATH 显式指定。
_KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMFY = os.environ.get("COMFYUI_PATH") or os.path.dirname(os.path.dirname(_KIT_DIR))
if not os.path.isdir(os.path.join(_COMFY, "comfy")):
    sys.stderr.write(
        "\n[FAIL] 找不到 ComfyUI 根目录（试过：%s）\n"
        "       请把本包装在 <ComfyUI>/custom_nodes/ 下，或设环境变量：\n"
        "       COMFYUI_PATH=/path/to/ComfyUI python tests/test_relay_core.py\n\n"
        % _COMFY)
    raise SystemExit(2)
sys.path.insert(0, _COMFY)
sys.path.insert(0, _KIT_DIR)

import comfy.nested_tensor as NT  # noqa: E402

import relay_core as CORE  # noqa: E402


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (("  " + detail) if detail else ""))


def expect_raise(name, fn, needle=""):
    try:
        fn()
    except Exception as e:
        ok = (needle in str(e)) if needle else True
        check(name, ok, "→ %s: %s" % (type(e).__name__, str(e).splitlines()[0][:90]))
        return
    check(name, False, "→ 未抛异常（应当 raise）")


def make_latent(frames, width=768, height=448, seed=0):
    """合成一段 H3 AV latent：视频 [1,24,T,28,48]、音频 [1,32,2,~] 。"""
    steps = CORE.steps_for_frames(frames)
    assert steps is not None, "%d 帧不是合法网格窗口" % frames
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(1, 24, steps, height // 16, width // 16, generator=g)
    at = int(round(CORE.FRAME_RESCALE * frames))
    a = torch.randn(1, 32, 2, at, generator=g)
    return {"samples": NT.NestedTensor([v, a])}, v, a


print("=" * 78)
print("1) 时序网格自洽")
print("=" * 78)
check("pixel_frames(7) == 22", CORE.pixel_frames(7) == 22, "得到 %d" % CORE.pixel_frames(7))
check("pixel_frames(12) == 39", CORE.pixel_frames(12) == 39, "得到 %d" % CORE.pixel_frames(12))
check("steps_for_frames(22) == 7", CORE.steps_for_frames(22) == 7, "得到 %s" % CORE.steps_for_frames(22))
check("steps_for_frames(39) == 12", CORE.steps_for_frames(39) == 12, "得到 %s" % CORE.steps_for_frames(39))
check("steps_for_frames(192) == 57", CORE.steps_for_frames(192) == 57, "得到 %s" % CORE.steps_for_frames(192))
check("steps_for_frames(30) == 9（30 也在网格上）", CORE.steps_for_frames(30) == 9, "得到 %s" % CORE.steps_for_frames(30))
check("GUIDE_RUNS 是「配 17k+5 段长」的推荐集，非全部网格值",
      CORE.steps_for_frames(30) is not None and 30 not in CORE.GUIDE_RUNS)
check("snap_guide_run(35) == 22", CORE.snap_guide_run(35) == 22, "得到 %d" % CORE.snap_guide_run(35))
check("GUIDE_RUNS 全部落在网格上", not CORE.self_check(), str(CORE.self_check()))
check("step_offsets(7) 首位 0", CORE.step_offsets(7)[0] == 0, str(CORE.step_offsets(7)))

print()
print("=" * 78)
print("2) 尾段切片逐位正确")
print("=" * 78)
prev, pv, pa = make_latent(192, seed=1)
cur, cv, ca = make_latent(192, seed=2)

blocks, offsets, covered = CORE.video_tail_from_latent(prev, 22)
check("22 帧尾段切出 7 块", len(blocks) == 7, "得到 %d" % len(blocks))
check("覆盖帧数 == 22", covered == 22, "得到 %d" % covered)
check("锚位起点 == 0", offsets[0] == 0, str(offsets))
total = int(pv.shape[2])
start = total - 7
same = all(torch.equal(blocks[k], pv[:1, :, start + k:start + k + 1]) for k in range(7))
check("每块与源 latent 尾段逐位相同", same)
check("尾段最后一块 == 源 latent 最后一 token",
      torch.equal(blocks[-1], pv[:1, :, -1:]))

# 注意第 3 参是**源段总帧数**（外溢偏差对着总帧数量才有意义，见函数 docstring）
tail_a, rt, overhang, raw_steps, grid_off = CORE.audio_tail_from_latent(prev, 22, 192)
check("音频尾段步数 == ceil(22/24*40) = 37（拓宽到整步）", rt == 37, "得到 %d" % rt)
check("raw_steps 报告理论步数 36.67", abs(raw_steps - 22 / 24.0 * 40) < 1e-6,
      "得到 %.4f" % raw_steps)
check("音频尾段与源尾段逐位相同", torch.equal(tail_a, pa[:1, ..., int(pa.shape[-1]) - rt:]))
check("音频栅格外溢在容差内", abs(overhang) < 0.5, "overhang=%.3f" % overhang)
check("标准网格段 grid_off=False", grid_off is False)
_, rt7, _, raw7, _ = CORE.audio_tail_from_latent(prev, 7, 192)
check("off-grid 音频窗 7 帧 → 拓宽到 12 整步（11.67 向上）", rt7 == 12, "得到 %d" % rt7)
check("off-grid 报告理论步数 11.67", abs(raw7 - 7 / 24.0 * 40) < 1e-6, "得到 %.4f" % raw7)

print()
print("=" * 78)
print("3) 硬错误必须 raise")
print("=" * 78)
wide, _, _ = make_latent(192, width=928, height=1600, seed=3)
expect_raise("分辨率不一致 → raise",
             lambda: CORE.plan_relay(cur, wide, 22), "无法缩放")
expect_raise("非推荐窗口 30 帧 → raise（不静默吸附）",
             lambda: CORE.plan_relay(cur, prev, 30), "整步续接窗口")
lat124, l124v, _ = make_latent(124, seed=5)
expect_raise("窗口 >= 段长 → raise",
             lambda: CORE.plan_relay(lat124, lat124, 124), "没有新内容")
check("未接 context → 直通（applied=False）",
      not CORE.plan_relay(cur, None, 22).applied)

# 短段：128 帧(38 步) 取 22 帧尾段 → start=31, 31%5=1 → 必须 raise
short, _, _ = make_latent(124, seed=4)
st = int(short["samples"].tensors[0].shape[2])
print("      （124 帧 = %d 步；取 22 帧尾段 start=%d, %%5=%d）" % (st, st - 7, (st - 7) % 5))
if (st - 7) % 5 == 0:
    ok = CORE.plan_relay(cur, short, 22).applied
    check("124 帧上取 22 帧尾段（周期对齐）→ 通过", ok)
else:
    expect_raise("124 帧上取 22 帧尾段（周期错位）→ raise",
                 lambda: CORE.plan_relay(cur, short, 22), "周期位置")

print()
print("=" * 78)
print("4) conditioning 注入")
print("=" * 78)
import node_helpers  # noqa: E402

MARK = 7.0
cond = [[torch.zeros(1, 8), {"minimax_keyframes": [
    {"resolved_frame_index": 0, "latent": torch.full((1, 24, 1, 28, 48), MARK)},
    {"resolved_frame_index": 191, "latent": torch.ones(1, 24, 1, 28, 48)},
]}]]
plan = CORE.plan_relay(cur, prev, 22)
out = CORE.apply_relay(cond, plan)
kfs = out[0][1]["minimax_keyframes"]
check("keyframes 数量 = 保留 1 + 新增 7", len(kfs) == 8, "得到 %d" % len(kfs))
f0 = [k for k in kfs if int(k["resolved_frame_index"]) == 0]
check("钉住区内的旧锚（标记 latent）被丢弃，只剩续接块自带的 frame 0",
      len(f0) == 1 and float(f0[0]["latent"].max()) != MARK,
      "frame0 锚数=%d" % len(f0))
check("末帧锚（frame 191）被保留",
      any(int(k["resolved_frame_index"]) == 191 for k in kfs))
check("音频 ref 已追加到 minimax_refs",
      len(out[0][1].get("minimax_refs") or []) == 1
      and out[0][1]["minimax_refs"][0]["kind"] == "audio")
check("直通路径不改 conditioning",
      CORE.apply_relay(cond, CORE.plan_relay(cur, None, 22)) is cond)

print()
print("=" * 78)
print("5) AV latent 落盘往返")
print("=" * 78)
tmp = os.path.join(tempfile.gettempdir(), "relay_kit_test", "stage_00000.safetensors")
CORE.save_av_latent(prev, tmp, note="unit test")
back = CORE.load_av_latent(tmp)
bv, ba = CORE.streams_from_latent(back)
check("视频流往返逐位相同", torch.equal(bv, pv))
check("音频流往返逐位相同", torch.equal(ba, pa))
check("往返后仍是 NestedTensor", hasattr(back["samples"], "unbind"))
check("往返后可再次切尾段",
      all(torch.equal(blocks[k], bv[:1, :, start + k:start + k + 1]) for k in range(7)))
print("      " + CORE.describe_latent(back))

print()
print("=" * 78)
print("6) 裁头部重叠（H3RelayTrimAV 的底层：视频音频同裁）")
print("=" * 78)
SR = 32000
N_FRAMES = 73
N_SAMPLES = int(round(N_FRAMES / 24.0 * SR))
imgs = torch.arange(N_FRAMES * 2 * 2 * 3, dtype=torch.float32).reshape(N_FRAMES, 2, 2, 3)
wave = torch.arange(2 * N_SAMPLES, dtype=torch.float32).reshape(1, 2, N_SAMPLES)
aud = {"waveform": wave, "sample_rate": SR}

t_imgs = CORE.trim_head_frames(imgs, 22)
t_aud = CORE.trim_audio_head(aud, 22, 24.0)
check("画面 73 → 51 帧", int(t_imgs.shape[0]) == 51, "得到 %d" % int(t_imgs.shape[0]))
check("裁后首帧 == 原第 22 帧（逐位）", torch.equal(t_imgs[0], imgs[22]))
check("裁后尾帧 == 原尾帧", torch.equal(t_imgs[-1], imgs[-1]))
check("音频 97333 → 68000 采样点",
      int(t_aud["waveform"].shape[-1]) == N_SAMPLES - int(round(22 / 24.0 * SR)),
      "得到 %d" % int(t_aud["waveform"].shape[-1]))
check("裁后音画时长一致（51 帧 == 68000 点 @32k）",
      abs(int(t_aud["waveform"].shape[-1]) / SR - 51 / 24.0) < 1e-6,
      "%.4fs" % (int(t_aud["waveform"].shape[-1]) / SR))
check("音频裁后首点 == 原第 29333 点（逐位）",
      float(t_aud["waveform"][0, 0, 0]) == float(wave[0, 0, int(round(22 / 24.0 * SR))]))
check("trim=0 → 画面原样返回", CORE.trim_head_frames(imgs, 0) is imgs)
check("trim=0 → 音频原样返回", CORE.trim_audio_head(aud, 0, 24.0) is aud)
check("audio=None → 返回 None（不炸）", CORE.trim_audio_head(None, 22, 24.0) is None)
check("采样率原样保留", int(t_aud["sample_rate"]) == SR)
expect_raise("裁的帧数 >= 段长 → raise（裁完没画面）",
             lambda: CORE.trim_head_frames(imgs, 73), "裁完就没画面")
expect_raise("裁的采样点 >= 音频长度 → raise",
             lambda: CORE.trim_audio_head(aud, 999, 24.0), "音频只有")

print()
print("=" * 78)
print("7) 节点返回值契约（每个分支的返回路数必须 == len(RETURN_TYPES)）")
print("=" * 78)
import importlib.util  # noqa: E402
import types  # noqa: E402

# 目录名含连字符，不能直接当包名 → 伪造一个包壳再按文件加载 nodes.py
_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("h3relay_kit")
_pkg.__path__ = [_KIT]
sys.modules["h3relay_kit"] = _pkg
_spec = importlib.util.spec_from_file_location("h3relay_kit.nodes", os.path.join(_KIT, "nodes.py"))
NODES = importlib.util.module_from_spec(_spec)
sys.modules["h3relay_kit.nodes"] = NODES
_spec.loader.exec_module(NODES)


def arity(cls, kw):
    out = getattr(cls(), cls.FUNCTION)(**kw)
    return len(out) == len(cls.RETURN_TYPES), out


ok, out = arity(NODES.H3RelayMotionContext,
                dict(conditioning=cond, latent=cur, trim_frames=22, context_latent=None))
check("MotionContext 直通分支返回 3 路（曾漏第 3 路 → list index out of range）", ok, "实得 %d" % len(out))
check("直通分支 trim_frames 输出 = 0（首段不裁）", int(out[2]) == 0, "得到 %r" % (out[2],))

ok, out = arity(NODES.H3RelayMotionContext,
                dict(conditioning=cond, latent=cur, trim_frames=22, context_latent=prev))
check("MotionContext 续接分支返回 3 路", ok, "实得 %d" % len(out))
check("续接分支 trim_frames 输出 = 22（供裁剪节点）", int(out[2]) == 22, "得到 %r" % (out[2],))

img73 = torch.zeros(73, 2, 2, 3)
ok, out = arity(NODES.H3RelayTrimAV, dict(images=img73, trim_frames=0, fps=24.0, audio=None))
check("TrimAV trim=0 分支返回 3 路", ok, "实得 %d" % len(out))
ok, out = arity(NODES.H3RelayTrimAV, dict(images=img73, trim_frames=22, fps=24.0, audio=aud))
check("TrimAV 裁剪分支返回 3 路", ok, "实得 %d" % len(out))
check("裁剪分支画面 73 → 51 帧", int(out[0].shape[0]) == 51, "得到 %d" % int(out[0].shape[0]))
check("裁剪分支音频与画面同裁（68000 点）",
      int(out[1]["waveform"].shape[-1]) == 68000, "得到 %d" % int(out[1]["waveform"].shape[-1]))

ok, out = arity(NODES.H3RelayLatentSave,
                dict(latent=prev, run_id="unittest_arity", stage_index=0, note="arity test"))
check("LatentSave 返回 2 路", ok, "实得 %d" % len(out))
ok, out = arity(NODES.H3RelayLatentLoad,
                dict(run_id="unittest_arity", stage_index=1, explicit_path=""))
check("LatentLoad 返回 2 路（stage_index=本段号，读的是上一段）", ok, "实得 %d" % len(out))
check("LatentLoad 读回上一段 latent（192 帧 @768x448）",
      CORE.pixel_frames(int(out[0]["samples"].tensors[0].shape[2])) == 192)

# ---------------------------------------------------------------- 第 8 组：接缝自检
print()
print("[8] 接缝自检 find_head_jump / describe_head_jump")

# 8.1 干净起点：全段缓慢变化（帧差稳定在 3 左右）→ 不应报突变
seq_clean = torch.zeros(30, 8, 8, 3)
for i in range(30):
    seq_clean[i] = (i % 3) * 10.0        # 帧差恒定 ≈ 小幅
j, jump, base = CORE.find_head_jump(seq_clean)
check("8.1 平稳序列无突变（j<0）", j < 0, "j=%d jump=%.2f base=%.2f" % (j, jump, base))

# 8.2 首帧突变（模拟实测：第 0 帧独立、之后稳定）→ 应报 j=0
seq_jump = torch.zeros(30, 8, 8, 3)
for i in range(1, 30):
    seq_jump[i] = 50.0                    # 第 1 帧起基本一致
j, jump, base = CORE.find_head_jump(seq_jump)
check("8.2 首帧突变被检出（j=0）", j == 0, "j=%d jump=%.2f base=%.2f" % (j, jump, base))
check("8.2 突变比值超过阈值", jump > CORE.JUMP_RATIO * max(base, CORE._baseline_floor(seq_jump)),
      "jump=%.1f base=%.2f" % (jump, base))

# 8.3 中段突变（模拟实测：第 22→23 帧跳）→ 应报 j=22
seq_mid = torch.zeros(40, 8, 8, 3)
for i in range(1, 23):
    seq_mid[i] = 50.0 + (i % 2)
for i in range(23, 40):
    seq_mid[i] = 200.0 + (i % 2)
j, jump, base = CORE.find_head_jump(seq_mid)
check("8.3 第 22→23 帧突变被检出（j=22）", j == 22, "j=%d jump=%.2f base=%.2f" % (j, jump, base))

# 8.4 报告文案：有突变时给"再加 N 帧"的建议
msg = CORE.describe_head_jump(seq_mid)
check("8.4 突变报告含「建议 trim 再加 23 帧」", "23 帧" in msg, msg[:90])
msg2 = CORE.describe_head_jump(seq_clean)
check("8.4 平稳报告含「起点干净」", "起点干净" in msg2, msg2[:90])

# 8.5 帧数过少不应崩
j, jump, base = CORE.find_head_jump(torch.zeros(3, 8, 8, 3))
check("8.5 帧数过少返回 (-1,0,0) 不抛异常", j == -1, "j=%d" % j)

# ---------------------------------------------------------------- 第 9 组：段号声明与取源的矛盾必须硬拦
print()
print("[9] stage_index 声明了第 N 段却拿不到上一段 → 必须 raise（不得静默直通）")

ctx = NODES.H3RelayMotionContext()

# 9.1 stage_index>=1 + run_id 空 + 无 context_latent → 旧行为是静默直通（坏片），必须 raise
try:
    ctx.apply(cond, cur, trim_frames=22, context_latent=None, audio_frames=0,
              run_id="", stage_index=1)
    check("9.1 段号≥1 且无来源 → raise", False, "竟然没报错（会产出无续接的哑剧）")
except RuntimeError as e:
    check("9.1 段号≥1 且无来源 → raise", "静默直通" in str(e), str(e).split("\n")[0])
except Exception as e:  # noqa: BLE001
    check("9.1 段号≥1 且无来源 → raise", False, "抛的是 %s：%s" % (type(e).__name__, e))

# 9.2 run_id 只有空白也算空
try:
    ctx.apply(cond, cur, trim_frames=22, context_latent=None, audio_frames=0,
              run_id="   ", stage_index=3)
    check("9.2 run_id 全空白同样 raise", False, "竟然没报错")
except RuntimeError as e:
    check("9.2 run_id 全空白同样 raise", "静默直通" in str(e), str(e).split("\n")[0])

# 9.3 stage_index=0 仍应正常直通（独立段，合法）
try:
    out0 = ctx.apply(cond, cur, trim_frames=22, context_latent=None, audio_frames=0,
                     run_id="", stage_index=0)
    check("9.3 stage_index=0 仍直通不报错（独立段合法）", int(out0[2]) == 0, "trim=%r" % (out0[2],))
except Exception as e:  # noqa: BLE001
    check("9.3 stage_index=0 仍直通不报错（独立段合法）", False, "抛了 %s" % type(e).__name__)

# 9.4 手动接了 context_latent 时，段号≥1 不应被拦（高级用法）
try:
    out1 = ctx.apply(cond, cur, trim_frames=22, context_latent=prev, audio_frames=0,
                     run_id="", stage_index=1)
    check("9.4 手动接 context_latent → 段号≥1 不拦", int(out1[2]) == 22, "trim=%r" % (out1[2],))
except Exception as e:  # noqa: BLE001
    check("9.4 手动接 context_latent → 段号≥1 不拦", False, "抛了 %s：%s" % (type(e).__name__, e))

# 9.5 run_id 有值但文件不存在 → FileNotFoundError（不是静默直通）
try:
    ctx.apply(cond, cur, trim_frames=22, context_latent=None, audio_frames=0,
              run_id="unittest_no_such_run", stage_index=1)
    check("9.5 run_id 有值但无文件 → FileNotFoundError", False, "竟然没报错")
except FileNotFoundError:
    check("9.5 run_id 有值但无文件 → FileNotFoundError", True)
except Exception as e:  # noqa: BLE001
    check("9.5 run_id 有值但无文件 → FileNotFoundError", False, "抛的是 %s" % type(e).__name__)

# ---------------------------------------------------------------- 第 10 组：latent 取流对非 NestedTensor 的健壮性
print()
print("[10] streams_from_latent：不能把普通张量当 NestedTensor 拆 batch 维")

# 10.1 NestedTensor → 按 .tensors 拆成两条流
nt_latent = {"samples": NT.NestedTensor([torch.zeros(1, 24, 57, 48, 28),
                                         torch.zeros(1, 32, 2, 320)])}
sp = CORE.streams_from_latent(nt_latent)
check("10.1 NestedTensor → 2 条流（视频+音频）", len(sp) == 2 and sp[0].shape[1] == 24,
      "得到 %d 条" % len(sp))

# 10.2 普通 [B,C,T,H,W] 张量（B=1）→ 必须当成**一条**流，不能沿 batch 维拆
plain1 = {"samples": torch.zeros(1, 24, 57, 48, 28)}
sp1 = CORE.streams_from_latent(plain1)
check("10.2 普通张量 B=1 → 1 条流（不是被 unbind 成 1 条 4 维）",
      len(sp1) == 1 and sp1[0].ndim == 5, "得到 %d 条，ndim=%d" % (len(sp1), sp1[0].ndim))

# 10.3 B=4 → 仍然只能是一条流（旧写法会拆成 4 条"伪音频流"）
plain4 = {"samples": torch.zeros(4, 24, 57, 48, 28)}
sp4 = CORE.streams_from_latent(plain4)
check("10.3 普通张量 B=4 → 仍是 1 条流（旧写法会错拆成 4 条）",
      len(sp4) == 1 and sp4[0].shape[0] == 4, "得到 %d 条" % len(sp4))

# 10.4 video-only latent 去当 context_latent → 必须明确 raise，不能静默当"有音频"
try:
    CORE.audio_from_latent(plain1)
    check("10.4 video-only → audio_from_latent 必须 raise", False, "竟然没报错")
except ValueError as e:
    check("10.4 video-only → audio_from_latent 必须 raise", "没有音频流" in str(e), str(e).split("\n")[0])
except Exception as e:  # noqa: BLE001
    check("10.4 video-only → audio_from_latent 必须 raise", False, "抛的是 %s" % type(e).__name__)

# 10.5 list 形态（已拆开的流）仍可用
sp_list = CORE.streams_from_latent([torch.zeros(1, 24, 57, 48, 28), torch.zeros(1, 32, 2, 320)])
check("10.5 list 形态仍拆成 2 条流", len(sp_list) == 2, "得到 %d 条" % len(sp_list))

# 10.6 完全不是张量的输入 → 明确报错
try:
    CORE.streams_from_latent({"samples": "not a tensor"})
    check("10.6 非张量输入 → raise", False, "竟然没报错")
except ValueError:
    check("10.6 非张量输入 → raise", True)

# ---------------------------------------------------------------- 第 11 组：Chain 的状态格子
print()
print("[11] H3RelayChain：前端要往 status 写状态，后端必须有这一格")

chain_cls = NODES.H3RelayChain
req = chain_cls.INPUT_TYPES().get("required") or {}
opt = chain_cls.INPUT_TYPES().get("optional") or {}
check("11.1 status 作为可选 widget 存在（否则前端提示无处显示）", "status" in opt,
      "optional=%s" % list(opt))
check("11.2 status 排在最后一个（旧工作流少这一格也不会让前面取值错位）",
      list(req) == ["segments"] and list(opt)[-1] == "status",
      "required=%s optional=%s" % (list(req), list(opt)))

# 11.3 前端会把 status 一起传进来 → noop 必须能吃下
try:
    chain_cls().noop(segments=3, status="第 2 段")
    check("11.3 noop 能吃下 status 参数（不会 TypeError）", True)
except Exception as e:  # noqa: BLE001
    check("11.3 noop 能吃下 status 参数（不会 TypeError）", False, repr(e))

# 11.4 前端 JS 确实在找这个 widget 名
_js = os.path.join(_KIT_DIR, "web", "relay_kit_chain.js")
try:
    _src = open(_js, encoding="utf-8").read()
    check("11.4 前端 JS 找的 widget 名与后端一致（status）",
          'x.name === "status"' in _src, "未在 relay_kit_chain.js 里找到该查找")
except Exception as e:  # noqa: BLE001
    check("11.4 前端 JS 找的 widget 名与后端一致（status）", False, repr(e))

# ---------------------------------------------------------------- 第 12 组：沉降帧
print()
print("[12] 沉降帧：pin 与 crop 解耦 + 自动检测（观测端决定，用户零配置）")


def seam_seg(n, switch_at, level=200.0, cut_every=None):
    """合成一段"续接段 decode 结果"。

    [0, switch_at) = 上一段尾部的**复现**（帧差 ≈ 0~3）；[switch_at, n) = 本段新内容
    （基准亮度不同 → 切换处 1 帧硬跳）。cut_every 用来塞一个"段内真实切镜"。
    """
    im = torch.zeros(n, 8, 8, 3)
    for i in range(n):
        im[i] = (0.0 if i < switch_at else level) + (i % 3) * 3.0
    if cut_every:
        for i in range(cut_every, n, cut_every):
            im[i] = 250.0 - (i % 5)
    return im


# 12.1~12.5 core 层：settle 只动 crop，不动 pin / 锚位 / 音频窗
pl0 = CORE.plan_relay(cur, prev, 22)
check("12.1 settle=0 → trim == 22 且 settle == 0（旧行为逐位不变，回归保护）",
      pl0.trim == 22 and pl0.settle == 0, "trim=%d settle=%d" % (pl0.trim, pl0.settle))
check("12.1b summary() 正常格式化（带沉降字段）", "裁首 22 帧（含沉降 0）" in pl0.summary(), pl0.summary())
pl3 = CORE.plan_relay(cur, prev, 22, settle_frames=3)
check("12.2 settle=3 → trim == 25，但 span 仍 22、锚位不变（pin 未被污染）",
      pl3.trim == 25 and pl3.span == 22 and pl3.indices == pl0.indices,
      "trim=%d span=%d" % (pl3.trim, pl3.span))
check("12.3 settle 不影响音频窗（仍 37 步）",
      pl3.audio_ref["ref_audio_t"] == pl0.audio_ref["ref_audio_t"] == 37)
check("12.4 settle<0 → 归零，不报错",
      CORE.plan_relay(cur, prev, 22, settle_frames=-5).trim == 22)
expect_raise("12.5 pin + settle >= 段长 → raise（裁完没画面）",
             lambda: CORE.plan_relay(lat124, lat124, 22, settle_frames=103), "裁完就没画面")

# 12.6~12.10 检测器：窄窗 + 上限 + 失败安全
sA, _, _ = CORE.detect_settle(seam_seg(107, 15), 22)
check("12.6 切换点(15) 落在钉住区之内 → settle=0（不该多裁）", sA == 0, "settle=%d" % sA)
sB, jB, _ = CORE.detect_settle(seam_seg(73, 23), 22)
check("12.7 切换点在原第 22→23 帧 → settle=1（正是 0.1.1 实测的那个场景）",
      sB == 1, "settle=%d jump=%.1f" % (sB, jB))
sC, _, _ = CORE.detect_settle(seam_seg(73, 23, cut_every=25), 22)
check("12.8 段内第 25 帧有真实切镜 → 仍取 1，不把新内容裁掉（D3 回归）",
      0 <= sC <= CORE.MAX_SETTLE, "settle=%d" % sC)
sD, _, _ = CORE.detect_settle(seam_seg(120, 40), 22)
check("12.9 切换点远超 pin+MAX_SETTLE → 回退 0（宁可维持旧行为也不赌）", sD == 0, "settle=%d" % sD)
check("12.10 pin<=0 时不检测（首段/独立段绝不触发）",
      CORE.detect_settle(seam_seg(73, 23), 0) == (0, 0.0, 0.0))

# 12.11~12.13 自检文案：只有**可执行**的才给建议（D3）
msg_far = CORE.describe_head_jump(seq_mid)      # 突变在裁后第 22 帧 → 太远
check("12.11 突变位置超出可执行范围 → 不再建议「再裁」（D3）",
      "不动刀" in msg_far and "settle_frames" not in msg_far, msg_far[:100])
check("12.12 但仍然报出位置（含「第 22→23 帧」）", "第 22→23 帧" in msg_far)
seq_near = torch.zeros(30, 8, 8, 3)
for i in range(3, 30):
    seq_near[i] = 200.0
msg_near = CORE.describe_head_jump(seq_near)    # 突变就在裁后头部 → 可执行
check("12.13 突变就在头部 → 给出可执行的 settle_frames 建议", "settle_frames" in msg_near, msg_near[:90])

# 12.14~12.22 裁节点：默认自动、可关、可固定
seg73 = seam_seg(73, 23)
aud73 = {"waveform": torch.zeros(1, 2, int(round(73 / 24.0 * SR))), "sample_rate": SR}
ok, out = arity(NODES.H3RelayTrimAV, dict(images=seg73, trim_frames=22, fps=24.0, audio=aud73))
check("12.14 裁节点默认 settle_frames=-1 → 自动裁 23 帧（73 → 50）",
      int(out[0].shape[0]) == 50, "得到 %d 帧" % int(out[0].shape[0]))
check("12.15 自动裁后首帧 == 原第 23 帧（切换点被裁掉，逐位）",
      torch.equal(out[0][0], seg73[23]))
check("12.16 自动裁后音频同裁 23 帧 → 音画时长一致（容差 = 1 个采样点）",
      abs(int(out[1]["waveform"].shape[-1]) / SR - 50 / 24.0) < 1.0 / SR,
      "%.4fs" % (int(out[1]["waveform"].shape[-1]) / SR))
check("12.17 报告里写清「钉住 + 沉降」两段账", "钉住 22 + 沉降 1" in out[2], out[2].splitlines()[0][:80])
check("12.18 自动裁后自检「起点干净」（接缝真的修好了）",
      "起点干净" in out[2], out[2].splitlines()[-1][:90])
ok, out0 = arity(NODES.H3RelayTrimAV, dict(images=seg73, trim_frames=0, fps=24.0, audio=None))
check("12.19 trim_frames=0（首段/独立段）→ 原样返回，绝不触发自动检测",
      out0[0] is seg73 and int(out0[0].shape[0]) == 73)
ok, outx = arity(NODES.H3RelayTrimAV, dict(images=seg73, trim_frames=22, fps=24.0,
                                          audio=None, settle_frames=0))
check("12.20 settle_frames=0 → 回到 0.2.x 旧行为（73 → 51 帧）",
      int(outx[0].shape[0]) == 51, "得到 %d" % int(outx[0].shape[0]))
ok, outm = arity(NODES.H3RelayTrimAV, dict(images=seg73, trim_frames=22, fps=24.0,
                                          audio=None, settle_frames=9))
check("12.21 settle_frames=9（手动固定）→ 裁 31 帧，供各段等长用",
      int(outm[0].shape[0]) == 42, "得到 %d" % int(outm[0].shape[0]))
_req = NODES.H3RelayTrimAV.INPUT_TYPES()["required"]
_opt = NODES.H3RelayTrimAV.INPUT_TYPES()["optional"]
check("12.22 settle_frames 追加在 optional 末位（旧工作流少这一格不前移）",
      list(_opt)[-1] == "settle_frames" and list(_req) == ["images", "trim_frames", "fps"],
      "required=%s optional=%s" % (list(_req), list(_opt)))

# ============ 组13：模糊型沉降（v0.3.1）——帧差法盲区的高频能量补判 ============

def blur_seg(n=40, pin=22, blur=(22, 28), amp=20.0, dev=15.0, seed=7):
    """模糊型沉降合成。钉住区 = 纹理A（复现，帧差中等）；[blur0,blur1) = 重绘发虚
    （常数帧：锐度≈0、帧差≈0）；其后 = 纹理B（新内容，均值同、振幅略小）。
    边界帧差刻意 < 4×基线 → 帧差法必然盲，只有锐度法能救。"""
    g = torch.Generator().manual_seed(seed)
    texA = 100.0 + torch.rand(8, 8, 3, generator=g) * amp
    texB = 100.0 + amp / 2 + (torch.rand(8, 8, 3, generator=g) - 0.5) * 2 * dev
    base_mean = float(texA.mean())
    im = torch.zeros(n, 8, 8, 3)
    for i in range(n):
        if i < pin:
            im[i] = texA + (i % 3) * 3.0
        elif blur[0] <= i < blur[1]:
            im[i] = base_mean
        else:
            im[i] = texB + (i % 3) * 3.0
    return im


sE, jE, bE = CORE.detect_settle(blur_seg(), 22)
check("13.1 模糊沉降（塌陷 f24-29、恢复 f30）→ 锐度法量出 settle≈8-9",
      6 <= sE <= CORE.MAX_SETTLE, "settle=%d dip=%.1f ref=%.1f" % (sE, jE, bE))
check("13.2 模糊路径的返回值语义：val=塌陷谷底（<0.35×基准）、base=复现区锐度基准",
      jE < CORE.BLUR_COLLAPSE_RATIO * bE and bE > 0.0,
      "dip=%.1f ref=%.1f" % (jE, bE))
sF, _, _ = CORE.detect_settle(blur_seg(blur=(22, 40)), 22)   # 糊过窗：22+18 > 22+12
check("13.3 塌陷越过检测窗且窗内无恢复 → 0（宁少勿多）", sF == 0, "settle=%d" % sF)
sG, _, _ = CORE.detect_settle(blur_seg(blur=(22, 22)), 22)   # 无模糊区
check("13.4 全程带纹理、无塌陷 → 0（不误伤）", sG == 0, "settle=%d" % sG)
sH, _, _ = CORE.detect_settle(blur_seg(blur=(22, 34)), 22)   # 糊 22..33（12 帧贴满窗）、34 起新内容
check("13.5 长塌陷(12帧)窗内可见恢复 → settle=12（贴满上限）",
      sH == 12, "settle=%d" % sH)
sI, _, _ = CORE.detect_settle(blur_seg(seed=11), 22)         # 换 seed 回归
check("13.6 换随机种子结论稳定（6 ≤ settle ≤ 12）",
      6 <= sI <= CORE.MAX_SETTLE, "settle=%d" % sI)

# —— v0.3.2 双基准：复现区自身发软时，窗外新内容区才是真基准 ——

def blur_seg_mild(n=40, pin=22, blur=(22, 26), seed=9):
    """HD-mild 动机案例：复现区本身半软（振幅 20），塌陷区 0.3-0.5×源
    （对软复现基准不可见），新内容全锐（振幅 60）。
    0.3.1 单基准（复现区）→ dip>0.35×ref 返回 0；0.3.2 双基准 → 量出 settle。"""
    g = torch.Generator().manual_seed(seed)
    texA = 100.0 + torch.rand(8, 8, 3, generator=g) * 20.0
    texB = 110.0 + (torch.rand(8, 8, 3, generator=g) - 0.5) * 60.0
    meanA = float(texA.mean())
    im = torch.zeros(n, 8, 8, 3)
    for i in range(n):
        if i < pin:
            im[i] = texA + (i % 2) * 6.0
        elif blur[0] <= i < blur[1]:
            im[i] = 0.5 * texA + 0.5 * meanA + (i % 3) * 1.5
        else:
            im[i] = texB + (i % 3) * 3.0
    return im


sM, jM, bM = CORE.detect_settle(blur_seg_mild(), 22)
check("13.7 HD-mild：对软复现基准不可见 → 双基准量出 settle=4",
      sM == 4, "settle=%d dip=%.1f ref=%.1f" % (sM, jM, bM))
_imM = blur_seg_mild()
_shM = CORE._sharpness(_imM[:36])
_faM = CORE._sharpness(_imM[36:76])
_expM = max(float(_shM[:22].median()), float(_faM.median()))
check("13.8 双基准语义：ref = max(复现区锐度, 窗外新内容区锐度)",
      abs(bM - _expM) < 1e-4, "ref=%.2f expect=%.2f" % (bM, _expM))
# —— v0.3.2 量纲不变性：0-1 产线数据与 0-255 测试数据必须同一结论 ——

check("13.9 量纲不变（帧差路）：0-255 的 settle=1 在 0-1 数据上同为 1（BASELINE_FLOOR 量纲 bug 回归）",
      CORE.detect_settle(seam_seg(73, 23) / 255.0, 22)[0] == 1,
      "settle=%d" % CORE.detect_settle(seam_seg(73, 23) / 255.0, 22)[0])
check("13.10 量纲不变（锐度路）：0-1 的 HD-mild 同样量出 settle=4",
      CORE.detect_settle(blur_seg_mild() / 255.0, 22)[0] == 4,
      "settle=%d" % CORE.detect_settle(blur_seg_mild() / 255.0, 22)[0])

# —— v0.3.3 段体自参考 + 假跳否决 ——

def soft_uniform_seg(n=40, pin=22, seed=5):
    """全段一致偏软（4 步低清风格）：头与体都是弱纹理，整体量级低但无塌陷差。"""
    g = torch.Generator().manual_seed(seed)
    im = torch.zeros(n, 8, 8, 3)
    for i in range(n):
        base = 100.0 + torch.rand(8, 8, 3, generator=g) * (6.0 if i < pin else 9.0)
        im[i] = base + (i % 3) * 1.0
    return im


sS, _, _ = CORE.detect_settle(soft_uniform_seg(), 22)
check("13.11 全段一致偏软（整体量级低、头体无差）→ 0（量纲无关的决策）",
      sS == 0, "settle=%d" % sS)

def ramp_blur_seg(n=40, pin=22, seed=3):
    """渐进模糊：f22-25 纹理振幅按 t=0.4/0.55/0.7/0.85 递减到深塌陷，f26 起新内容。"""
    g = torch.Generator().manual_seed(seed)
    texA = 100.0 + torch.rand(8, 8, 3, generator=g) * 20.0
    texB = 110.0 + (torch.rand(8, 8, 3, generator=g) - 0.5) * 30.0
    meanA = float(texA.mean())
    im = torch.zeros(n, 8, 8, 3)
    ts = [0.4, 0.55, 0.7, 0.85]
    for i in range(n):
        if i < pin:
            im[i] = texA + (i % 3) * 3.0
        elif i < pin + 4:
            tt = ts[i - pin]
            im[i] = (1 - tt) * texA + tt * meanA
        else:
            im[i] = texB + (i % 3) * 3.0
    return im


sR, vR, _ = CORE.detect_settle(ramp_blur_seg(), 22)
check("13.12 渐进模糊（缓坡到底才过深线）→ settle=4（覆盖整个塌陷前缀）",
      sR == 4, "settle=%d" % sR)

def fake_jump_seg(n=40, pin=22, seed=4):
    """假跳：f22 既是深塌陷又是一次大跳（亮常帧），f23 起新内容——
    跳点必须被验身否决，由锐度路定界（返回值 = 塌陷谷底而非跳变帧差）。"""
    g = torch.Generator().manual_seed(seed)
    texA = 100.0 + torch.rand(8, 8, 3, generator=g) * 20.0
    texB = 110.0 + (torch.rand(8, 8, 3, generator=g) - 0.5) * 30.0
    im = torch.zeros(n, 8, 8, 3)
    for i in range(n):
        if i < pin:
            im[i] = texA + (i % 3) * 3.0
        elif i == pin:
            im[i] = 250.0
        else:
            im[i] = texB + (i % 3) * 3.0
    return im


sF, vF, bF = CORE.detect_settle(fake_jump_seg(), 22)
check("13.13 假跳被验身否决 → 锐度路定界 settle=1，信号=塌陷谷底（非跳变帧差）",
      sF == 1 and vF < 10.0 and bF > 100.0,
      "settle=%d val=%.1f ref=%.1f" % (sF, vF, bF))

# ============ 组14：拷贝桥（0.4.0）——latent 硬拷贝 + 噪声掩码 ============

def av_latent(t_steps, a_ticks=48, seed=0, w=8, h=8):
    g = torch.Generator().manual_seed(seed)
    video = torch.rand(1, 4, t_steps, h, w, generator=g)
    audio = torch.rand(1, 2, 2, a_ticks, generator=g)
    return {"samples": CORE._nested_pair(video, audio)}


tgt = av_latent(12, seed=1)
prv = av_latent(12, seed=2)
outC, trimC, repC = CORE.build_continue_latent(tgt, prv, 22)
sv = CORE.video_from_latent(outC)
pv = CORE.video_from_latent(prv)
tv = CORE.video_from_latent(tgt)
check("14.1 拷贝位级一致：前缀 7 步 == 上段尾部 7 步（start=5 对齐）",
      torch.equal(sv[:, :, :7], pv[:, :, 5:12]))
check("14.2 前缀之后保留本段原 latent", torch.equal(sv[:, :, 7:], tv[:, :, 7:]))
check("14.3 输入 target 未被变异", torch.equal(tv, CORE.video_from_latent(tgt)))
m = outC["noise_mask"]
check("14.4 hard 掩码：形状 [1,1,12,8,8]，前 7 步=0 其余=1，有限",
      tuple(m.shape) == (1, 1, 12, 8, 8) and bool((m[:, :, :7] == 0).all())
      and bool((m[:, :, 7:] == 1).all()) and bool(torch.isfinite(m).all()))
check("14.5 trim 输出 = covered = 22（接 TrimAV）", trimC == 22 and "22" in repC)
oa = CORE.audio_from_latent(outC)
pa = CORE.audio_from_latent(prv)
ta = CORE.audio_from_latent(tgt)
rt = int(oa.shape[-1] - round(22 / 24.0 * 40.0) + 0)  # 期望 37
check("14.6 音频尾拷贝：前 37 tick == 上段尾部，其后保留本段",
      torch.equal(oa[..., :37], pa[..., 11:48]) and torch.equal(oa[..., 37:], ta[..., 37:]))
outT, _, trimT = CORE.build_continue_latent(av_latent(12, seed=3), av_latent(12, seed=4), 22,
                                            mask_mode="taper")
wexp = CORE.prefix_taper_weights(7, 4, 0.10)
mT = outT["noise_mask"]
check("14.7 taper 掩码：头 1.0 线性降到缝端 0.10",
      torch.allclose(mT[:, :, :7, 0, 0], torch.tensor(wexp, dtype=torch.float32), atol=1e-6)
      and abs(float(mT[:, :, 6, 0, 0]) - 0.10) < 1e-6, "weights=%s" % (wexp,))
expect_raise("14.8 跨分辨率 → raise（拷贝桥禁止）",
             lambda: CORE.build_continue_latent(tgt, av_latent(12, seed=5, w=8, h=16), 22), "跨分辨率")
big_prev = av_latent(22, seed=6)
expect_raise("14.9 前缀占满整段（73f→22 步 ≥ 目标 12 步）→ raise",
             lambda: CORE.build_continue_latent(tgt, big_prev, 73), "没有新内容")
mis_prev = av_latent(13, seed=7)
expect_raise("14.10 尾段起点非 5 对齐 → raise（接缝位移防线复用）",
             lambda: CORE.build_continue_latent(tgt, mis_prev, 22), "起始于周期位置")
expect_raise("14.11 mask_mode 非法 → raise",
             lambda: CORE.build_continue_latent(tgt, prv, 22, mask_mode="soft"), "mask_mode")
okC, outN = arity(NODES.H3RelayCopyBridge, dict(latent=av_latent(12, seed=8),
                                                context_latent=av_latent(12, seed=9),
                                                context_frames=22))
check("14.12 节点返回 3 路（latent/report/trim）", okC, "实得 %d" % len(outN))
check("14.13 节点 trim 输出 = 22", int(outN[2]) == 22)
_it = NODES.H3RelayCopyBridge.INPUT_TYPES()
check("14.14 节点注册 + required 键序 + optional 末位（追加铁律）",
      "H3RelayCopyBridge" in NODES.NODE_CLASS_MAPPINGS
      and list(_it["required"]) == ["latent", "context_latent", "context_frames"]
      and list(_it["optional"])[-1] == "pin_audio",
      "required=%s optional=%s" % (list(_it["required"]), list(_it["optional"])))

# —— 14.15/14.16 掩码语义钉子（2026-09-15）——
# 动机：taper 曾被下游误当「软一点的硬锁」当默认跑了 23 次，段首被重画导致「续不上」。
# 这里把语义本身断言下来：hard 全 0 = 真钉住；taper 每一帧 m>0 = **没有被钉住**。
# 谁要改 taper 的方向，先过这两条，再回头改 nodes/README/CHANGES 的措辞。
check("14.15 hard 掩码 = 钉住区全 0（真钉住：d*0 + anchor*1）",
      bool((outC["noise_mask"][:, :, :7] == 0).all()),
      "min=%s max=%s" % (float(m[:, :, :7].min()), float(m[:, :, :7].max())))
check("14.16 taper 掩码 = 钉住区**无一处为 0**（头 1.0 全重绘，缝端仍留 seam_min）",
      bool((mT[:, :, :7] > 0).all()) and float(mT[:, :, 0, 0, 0]) == 1.0
      and float(mT[:, :, 6, 0, 0]) > 0.0,
      "头=%.2f 缝端=%.2f 最小=%.4f（taper 不钉住）"
      % (float(mT[:, :, 0, 0, 0]), float(mT[:, :, 6, 0, 0]), float(mT[:, :, :7].min())))

# ============ 组15：色档收敛信号（v0.4.1）——注噪/taper 收敛尾巴的观测端 ============

def grade_seg(n=40, pin=22, dark=(22, 25), factor=0.88, seed=13):
    """色档收敛合成：f22-24 整体压暗 12%（收敛中），f25 起回归体档。纹理均匀无模糊。"""
    g = torch.Generator().manual_seed(seed)
    texA = 100.0 + torch.rand(8, 8, 3, generator=g) * 20.0
    im = torch.zeros(n, 8, 8, 3)
    for i in range(n):
        f = factor if dark[0] <= i < dark[1] else 1.0
        im[i] = texA * f + (i % 3) * 3.0
    return im


sG, vG, bG = CORE.detect_settle(grade_seg(), 22)
check("15.1 色档收敛（暗 3 帧后回归体档）→ 锐度路不触发、色档路 settle=3",
      sG == 3, "settle=%d val=%.1f thr=%.1f" % (sG, vG, bG))

def wobble_seg(n=40, pin=22, seed=17):
    """自然亮度波动：体区各帧亮度随机 ±5%，头体同分布 → 不误裁。"""
    g = torch.Generator().manual_seed(seed)
    texA = 100.0 + torch.rand(8, 8, 3, generator=g) * 20.0
    im = torch.zeros(n, 8, 8, 3)
    for i in range(n):
        w = 1.0 + (torch.rand(1, generator=g).item() - 0.5) * 0.10
        im[i] = texA * w + (i % 3) * 3.0
    return im


sW, _, _ = CORE.detect_settle(wobble_seg(), 22)
check("15.2 自然亮度波动（头体同分布）→ 0（阈值随体 MAD 自适应）",
      sW == 0, "settle=%d" % sW)

sM2, _, _ = CORE.detect_settle(blur_seg_mild() * 0.85, 22)  # 模糊+整体压暗：锐度路与色档路同响
check("15.3 三信号取最大：mild 模糊(4) 与压暗色档(≥4) → settle ≥ 4",
      sM2 >= 4, "settle=%d" % sM2)
check("15.4 量纲不变：0-1 输入同结论",
      CORE.detect_settle(grade_seg() / 255.0, 22)[0] == 3,
      "settle=%d" % CORE.detect_settle(grade_seg() / 255.0, 22)[0])

# ============ 组16：0.4.2 回归（导出音频分支 / 中文 note / 服务端校验） ============

# 16.1 导出音频尾段分支（0.4.1 前返回 3 元组 → 解包必崩，此前从未被测）
exp_prev = dict(prev)
_pa2 = CORE.audio_from_latent(prev)     # 别用组 2 的 pa——组 14 已把同名变量覆盖成小 latent
exp_tail_v, _eo, _ec = CORE.video_tail_from_latent(prev, 22)
exp_prev[CORE.KEY_EXPORT_TAIL_VIDEO] = torch.cat(exp_tail_v, dim=2)
exp_prev[CORE.KEY_EXPORT_FRAMES] = 22
exp_prev[CORE.KEY_EXPORT_TAIL_AUDIO] = _pa2[0, :, :, -37:]   # 3 维，顺带测 unsqueeze
try:
    t_x, rt_x, oh_x, raw_x, go_x = CORE.audio_tail_from_latent(exp_prev, 22, 192)
    check("16.1 导出音频分支返回 5 元组且步数=尾段长", rt_x == 37 and go_x is False,
          "rt=%d" % rt_x)
except (ValueError, TypeError) as e:
    check("16.1 导出音频分支返回 5 元组且步数=尾段长", False, "→ %s" % e)
plan_x = CORE.plan_relay(cur, exp_prev, 22)
check("16.2 plan_relay 走导出尾段快路径不崩且续接成功",
      plan_x.applied and plan_x.audio_ref["ref_audio_t"] == 37)
cb_t, cb_trim, cb_rep = CORE.build_continue_latent(cur, exp_prev, 22)
check("16.3 拷贝桥走导出音频分支不崩（trim=22）", cb_trim == 22)

# 16.4 中文 note 落盘往返（0.4.1 前 ord(c)>255 直接崩）
tmp_c = os.path.join(tempfile.gettempdir(), "relay_kit_test", "stage_cjk.safetensors")
try:
    CORE.save_av_latent(prev, tmp_c, note="22帧窗 v2 备注——中文、emoji🎬")
    back_c = CORE.load_av_latent(tmp_c)
    # 比较基准现取：组 14 已把组 2 的局部名 pv/pa 覆盖成拷贝桥的小 latent
    check("16.4 中文 note 落盘往返不崩且流仍逐位相同",
          torch.equal(CORE.streams_from_latent(back_c)[0],
                      CORE.video_from_latent(prev)))
except Exception as e:
    check("16.4 中文 note 落盘往返不崩且流仍逐位相同", False, "→ %s: %s" % (type(e).__name__, e))
check("16.5 原子写：落盘后无 .tmp 残留", not os.path.isfile(tmp_c + ".tmp"))

# 16.6 stage_index=0 的友好报错可达（0.4.1 前被 _stage_path 抢抛「不能为负」）
expect_raise("16.6 LatentLoad stage_index=0 → 引导文案（而非「不能为负」）",
             lambda: NODES.H3RelayLatentLoad().load(run_id="unittest_arity", stage_index=0),
             "没有上一段可续")

# 16.7 run_id 非法字符替换为 _（不再静默同目录）
p_a = NODES._stage_path("my/film", 0)
p_b = NODES._stage_path("myfilm", 0)
check("16.7 run_id 含斜杠 → 替换为 _，与纯字母名不撞目录",
      p_a != p_b and "my_film" in p_a)
_nul_dir = os.path.basename(os.path.dirname(NODES._stage_path("NUL", 0)))
check("16.8 run_id Windows 保留名加后缀避让", _nul_dir == "NUL_", "目录名=%s" % _nul_dir)
expect_raise("16.9 run_id 全非法字符仍视为空 → raise",
             lambda: NODES._stage_path("///", 0), "不能为空")

# 16.10 fps 服务端校验（widget min=1 只挡 UI，API 可提交 0/NaN）
expect_raise("16.10 fps=0 → raise（不再 ZeroDivisionError）",
             lambda: NODES.H3RelayTrimAV().trim(images=img73, trim_frames=22, fps=0.0),
             "fps 必须")
expect_raise("16.11 fps=NaN → raise",
             lambda: NODES.H3RelayTrimAV().trim(images=img73, trim_frames=22, fps=float("nan")),
             "fps 必须")

# 16.12 音频栅格偏差告警可达（0.4.1 前 overhang 被覆写 0.0，if overhang: 永不触发）
off_grid = {"samples": NT.NestedTensor([
    CORE.video_from_latent(prev)[0],
    torch.zeros(1, 32, 2, 200),   # 音频 tick 远偏离 round(5/3*192)=320 → grid_off
])}
plan_g = CORE.plan_relay(cur, off_grid, 22)
check("16.12 非标准音频栅格 → notes 出现偏差告警",
      any("偏差超出半整步" in n for n in plan_g.notes),
      "notes=%s" % plan_g.notes)

# 16.13 pin_audio=False 走纯视频 latent（0.4.1 前无条件要求音频流）
pure_t = {"samples": torch.rand(1, 4, 12, 8, 8)}
pure_p = {"samples": torch.rand(1, 4, 12, 8, 8)}
try:
    out_p, trim_p, rep_p = CORE.build_continue_latent(pure_t, pure_p, 22, pin_audio=False)
    check("16.13 纯视频 latent + pin_audio=False 走通（trim=22）", trim_p == 22, rep_p)
except ValueError as e:
    check("16.13 纯视频 latent + pin_audio=False 走通（trim=22）", False, "→ %s" % e)
expect_raise("16.14 纯视频 latent + pin_audio=True → 明确 raise（需要音频流）",
             lambda: CORE.build_continue_latent(pure_t, pure_p, 22, pin_audio=True),
             "音频流")

# 16.15 契约降级不缓存：找不到上游时的放行不能被永久缓存
LC = NODES.CONTRACT        # layout_contract 是包内相对导入，走 nodes 已加载的模块对象
_saved = LC._CACHE
LC._CACHE = None
_orig_ext = LC._extract_frame_per_token
LC._extract_frame_per_token = lambda: (None, ["模拟：上游不可见"])
r1 = LC.check_layout()
calls = [0]
def _counting():
    calls[0] += 1
    return (None, ["模拟：上游不可见"])
LC._extract_frame_per_token = _counting
r2 = LC.check_layout()
LC._extract_frame_per_token = _orig_ext
LC._CACHE = _saved
check("16.15 降级放行不写缓存（下次执行会重试解析）",
      r1[0] and r2[0] and calls[0] == 1 and LC._CACHE is _saved)

# 16.16 拷贝桥掩码与 latent 同设备同 batch
mask_p = out_p["noise_mask"]
check("16.16 noise_mask 与 latent 同设备", mask_p.device == pure_t["samples"].device)
b_multi = {"samples": torch.rand(2, 4, 12, 8, 8)}
b_prev = {"samples": torch.rand(1, 4, 12, 8, 8)}
out_b, _, _ = CORE.build_continue_latent(b_multi, b_prev, 22, pin_audio=False)
check("16.17 batch=2 时掩码 batch 维随 target", int(out_b["noise_mask"].shape[0]) == 2)

print()
print("=" * 78)
print("结果：通过 %d / 失败 %d" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败项：")
    for f in FAIL:
        print("   -", f)
print("=" * 78)
sys.exit(1 if FAIL else 0)
