# -*- coding: utf-8 -*-
"""运行时契约：把本包的网格常量与 live ComfyUI 的 H3 上游源码对照。

我们所有的切片算术都建立在一个假设上：
    H3 的时序网格 FRAME_PER_TOKEN = (1, 4, 4, 4, 4)（帧 → latent token 的跨度）。
这个假设的真身定义在 ComfyUI 的 ``comfy/ldm/minimax/model.py``（上游节点
``comfy_extras/nodes_minimax_h3.py`` 是从那里 import 的）。ComfyUI 更新后
如果上游改了它，我们切出来的尾段就会整体错位，渲染出"看起来成功"的坏片子。

所以桥 / 裁节点第一次使用时把两边对照一遍：
    一致        → 通过，结果缓存到进程退出
    不一致      → 拒绝运行，说清两边各是什么
    找不到上游  → 放行但留痕提示（可能只是路径变了，不武断拦人）

本模块只做"读上游源码 + 对照"，不修改 ComfyUI 的任何东西。
"""
from __future__ import annotations

import importlib.util
import io
import os
import re
from typing import List, Optional, Tuple

from . import relay_core as CORE

_UPSTREAM_NODE = "comfy_extras.nodes_minimax_h3"

_CACHE: Optional[Tuple[bool, List[str]]] = None


def _read(path: str) -> Optional[str]:
    try:
        if os.path.isfile(path):
            return io.open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        pass
    return None


def _module_file(modname: str) -> Optional[str]:
    try:
        spec = importlib.util.find_spec(modname)
        if spec is not None and spec.origin and os.path.isfile(spec.origin):
            return spec.origin
    except Exception:
        pass
    # 兜底：ComfyUI 根目录 + 模块路径（comfy.ldm.minimax.model → comfy/ldm/...）
    try:
        import folder_paths
        root = folder_paths.base_path
    except Exception:
        root = os.getcwd()
    return os.path.join(root, *modname.split(".")) + ".py"


def _comfyui_root() -> str:
    try:
        import folder_paths
        return folder_paths.base_path
    except Exception:
        return os.getcwd()


def _extract_frame_per_token() -> Tuple[Optional[Tuple[int, ...]], List[str]]:
    """追着 import 链找 FRAME_PER_TOKEN 的真定义。返回 (定义, 过程消息)。"""
    msgs: List[str] = []
    root = _comfyui_root()

    # 1) 上游节点文件（定义可能直接在里面，也可能只是 import）
    node_src = _read(os.path.join(root, "comfy_extras", "nodes_minimax_h3.py"))
    if node_src is None:
        try:
            spec = importlib.util.find_spec(_UPSTREAM_NODE)
            node_src = _read(spec.origin) if spec and spec.origin else None
        except Exception:
            node_src = None
    if node_src is None:
        return None, ["未找到上游节点文件 comfy_extras/nodes_minimax_h3.py（路径变了？）"]
    msgs.append("已读上游节点 comfy_extras/nodes_minimax_h3.py")

    # 2) 直接定义？FRAME_PER_TOKEN = (…) / FRAME_PER_TOKEN: … = (…)
    m = re.search(
        r"FRAME_PER_TOKEN\s*(?::[^=\n]+)?=\s*[\[\(]\s*"
        r"([1-9][0-9]*(?:\s*,\s*[1-9][0-9]*)*)\s*[\]\)]",
        node_src,
    )
    if m:
        return tuple(int(x) for x in m.group(1).split(",")), msgs

    # 3) import 行 → 追到真定义文件（实测：from comfy.ldm.minimax.model import FRAME_PER_TOKEN）
    imp = re.search(
        r"from\s+(comfy[\w.]*)\s+import[^\n]*\bFRAME_PER_TOKEN\b",
        node_src,
    )
    if not imp:
        return None, msgs + ["上游节点里没有 FRAME_PER_TOKEN 的定义或 import（上游可能重构）"]
    model_file = _module_file(imp.group(1))
    model_src = _read(model_file or "")
    if model_src is None:
        return None, msgs + ["上游 import 自 %s，但读不到该文件" % imp.group(1)]
    msgs.append("定义在 %s" % (model_file or imp.group(1)))
    m = re.search(
        r"FRAME_PER_TOKEN\s*(?::[^=\n]+)?=\s*[\[\(]\s*"
        r"([1-9][0-9]*(?:\s*,\s*[1-9][0-9]*)*)\s*[\]\)]",
        model_src,
    )
    if not m:
        return None, msgs + ["在 %s 里没找到 FRAME_PER_TOKEN 的赋值定义" % imp.group(1)]
    return tuple(int(x) for x in m.group(1).split(",")), msgs


def check_layout(force: bool = False) -> Tuple[bool, List[str]]:
    """对照上游网格。返回 (是否放行, 消息列表)；结果按进程缓存。

    放行 ≠ 一定检查过：上游源码找不到时放行但带提示——
    真正危险的是"拿到了上游定义且不一致"，那种情况必须拒绝。
    """
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    msgs: List[str] = []
    ok = True
    upstream, msgs2 = _extract_frame_per_token()
    msgs.extend(msgs2)

    if upstream is None:
        msgs.append("契约检查未能确认上游网格（不影响本次运行，但请留意接缝表现）。")
        # ★ 不写缓存：「找不到上游」可能只是导入时序/路径问题。若把这次放行
        #   永久缓存，之后上游真的改了网格也再不会复查——下次执行会重试解析。
        return ok, msgs

    ours = tuple(CORE.FRAME_PER_TOKEN)
    if upstream == ours:
        msgs.append("契约通过：上游 FRAME_PER_TOKEN = %s，与本包一致。" % (ours,))
    else:
        ok = False
        msgs.append(
            "上游时序网格已变！上游 FRAME_PER_TOKEN = %s，本包 = %s。"
            "本包的尾段切片算术对上游失效，继续跑会产出错位的坏片子。"
            % (upstream, ours)
        )
    _CACHE = (ok, msgs)
    return _CACHE


def enforce() -> None:
    """供节点调用：契约不过就 raise，消息里说清两边各是什么。"""
    ok, msgs = check_layout()
    for m in msgs:
        print("[H3 Relay 契约] " + m)
    if not ok:
        raise RuntimeError(
            "H3 Relay 运行时契约失败：\n    " + "\n    ".join(msgs)
        )
