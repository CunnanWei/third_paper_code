# Repository Guidelines

本文件面向 `code/pretrain` 子仓库的贡献者，约定目录结构、开发与协作流程，帮助你快速对齐实践并减少返工。

## 项目结构与模块组织
- `modules/`：核心代码（`dataset.py`、`model.py`、`trainer.py`、`layers/`）。
- `scripts/`：可执行脚本（如 `train.py`）。
- `configs/config.py`：路径与训练超参（如 `batch_size`、`lr_rate`、`enable_wandb`）。
- `notebooks/`：数据预处理与实验脚本（如 `precompute_ecg_video.py`）。
- `checkpoints_pretrain/`：训练权重输出目录（自动创建）。
- 数据约定：数据集位于项目根目录下 `datasets/pretrain/`，CSV 包含 `path` 或 `video_path`、`total_report` 字段。

## 构建、测试与本地运行
- 创建环境（示例）：`python -m venv .venv && source .venv/bin/activate`。
- 安装依赖（最小集）：`pip install torch numpy pandas pyts tqdm wandb`。
- 预处理ECG视频：`python notebooks/precompute_ecg_video.py`（更新 CSV 并生成 `datasets/pretrain/ecg_video/*.npy`）。
- 训练：`python scripts/train.py`（内含训练与验证循环，权重保存到 `checkpoints_pretrain/`）。
- 关闭/开启 W&B：在 `configs/config.py` 设置 `enable_wandb=False/True`（或导出 `WANDB_DISABLED=true`）。

## 代码风格与命名
- Python 3.10+，缩进4空格；文件/函数/变量使用 `snake_case`，类用 `PascalCase`。
- 模块内导入顺序：标准库→第三方→本地；相对路径保持清晰、稳定。
- 建议添加简洁中文注释；函数应聚焦单一职责。

## 测试规范
- 目前未内置 `tests/`；建议使用 `pytest`，命名为 `tests/test_*.py`，运行 `pytest -q`。
- 优先覆盖：数据集取样形状、模型前向是否可执行、训练循环的基本收敛与权重落盘。

## Commit 与 Pull Request
- 推荐 Conventional Commits：`feat: ...`、`fix: ...`、`docs: ...`、`refactor: ...`、`test: ...`、`chore: ...`。
- PR 要求：简述动机与变更点、给出运行命令与关键日志/截图、关联 Issue、说明是否影响配置与数据路径。

## 安全与配置提示（可选）
- 请勿提交数据与大模型权重；建议将 `datasets/`、`checkpoints_pretrain/` 忽略在版本控制之外。
- 变更 `configs/config.py` 时，明确列出新增键的默认值与用途，避免破坏性更改。

