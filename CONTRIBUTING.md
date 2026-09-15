# 贡献指南

感谢你考虑为本项目贡献代码。

## 开发环境
- Python ≥ 3.10，需要 `torch` 与 `safetensors`。
- 把本目录放进 `ComfyUI/custom_nodes/` 即可被加载；纯算法层改动也可脱离 ComfyUI 单测。

## 离线自测（提 PR 前必须跑通）
```bash
python tests/test_relay_core.py
```
- 零 GPU、不加载模型、秒级完成。会自动上溯定位 ComfyUI 根目录；装在别处时用
  `COMFYUI_PATH=/path/to/ComfyUI python tests/test_relay_core.py` 指定。
  注意：脚本本身需要**能 import 到 ComfyUI**（`comfy.nested_tensor` / `node_helpers` /
  `folder_paths`）——`relay_core` 模块可独立复用，但这份打包测试**不能**脱离 ComfyUI 跑。
- 覆盖十六个方面（**实测执行 154 项断言**），是本包正确性的主要保障。
  报告里请贴实际执行数，不要按源码行数统计（互斥分支不会同时执行）。

## 代码纪律
- **硬错误必须 raise，绝不静默降级**（如帧数不在网格上、分辨率不匹配、段号与取源矛盾）。
- `relay_core.py` 是纯算法层，**不得 import 任何 ComfyUI 模块**，以保证可独立单测与复用。
- 新增 / 修改节点时，请同步更新 `README.md`（节点表 / 参数表 / 排障表）与 `CHANGES.md`。

## 提交信息
- 建议清晰说明「改了什么 / 为什么」，并关联相关 Issue。中英文均可。
- 提交请使用**非个人敏感**的 git 身份（例如 GitHub 提供的 no-reply 邮箱），避免把私人邮箱写进公开历史。
