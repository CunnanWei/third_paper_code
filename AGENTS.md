# Repository Guidelines

## 项目结构与模块组织
- 根目录 `code/`：核心实现，子目录 `pretrain/`（对比预训练）与 `finetune/`（若存在下游任务）。  
- `code/pretrain/modules/`：模型、数据集与自定义层；`scripts/train.py` 是预训练入口，支持单机多卡。  
- `datasets/`：按 `datasets/pretrain/{train,val}.csv` 及对应 `.npy/.npz` 存放输入；确保路径与 CSV 字段 `path`、`report_tokenize_path` 对齐。  
- `notebooks/`：如 `precompute_ecg_video.py`、`tokenize.py`，用于离线特征/标注处理。  
- `checkpoints_pretrain/`：训练权重输出目录，脚本首个 epoch 会创建 `ckpt_时间戳/` 并快照 `model.py`。

## 构建、测试与开发命令
- 进入虚拟环境：`python -m venv .venv && source .venv/bin/activate`。  
- 安装依赖：`pip install -r requirements.txt`（若无文件，可参考 `torch numpy pandas pyts tqdm wandb mamba-ssm transformers`）。  
- 预处理 ECG：`python code/pretrain/notebooks/precompute_ecg_video.py`。  
- 运行预训练：`python code/pretrain/scripts/train.py`（自动检测 GPU 数；如需自定义，使用 `torchrun --nproc_per_node=N ...`）。  
- 暂未定义统一测试命令，建议以 `pytest -q` 组织。

## 编码风格与命名约定
- 语言 Python 3.10+，4 空格缩进；模块/函数/变量使用 `snake_case`，类名 `PascalCase`。  
- 导入顺序：标准库 → 第三方 → 本地模块；相对路径保持稳定。  
- 注释尽量使用简体中文简述关键逻辑；禁止引入动态 import、`any` 强制转换及额外的防御性 try/except。  
- 不手动添加类型标注（遵循 AGENT 约束）。

## 测试准则
- 优先覆盖：数据加载形状、`ECGPretrainModel` 正向传播、训练循环保存权重。  
- 测试命名 `tests/test_*.py`，基于 `pytest`；可用小批量假数据或 `torch.randn` 构造输入。  
- 若使用 DDP，测试时可将 `Config.enable_distributed=False`，降低资源需求。

## Commit 与 Pull Request 指南
- 建议使用 Conventional Commits，如 `feat: add grad scaler`、`fix: handle sampler seed`。  
- PR 描述需包含：变更动机、关键修改点、运行命令与日志片段、涉及的数据/配置调整，并关联相关 Issue。  
- 若修改训练脚本或配置，说明对吞吐、显存及收敛的影响，并附上必要截图（W&B、loss 曲线等）。

## 安全与配置提示
- 不要将 `datasets/` 与大体积检查点提交到 Git；可在 `.gitignore` 保持忽略。  
- 若启用多卡，确保 `Config.enable_distributed=True` 且 CUDA 驱动已正确配置；单卡调试时保持 False，避免 mp.spawn。  
- 机密凭据（如 W&B key）请用环境变量注入，不要硬编码到配置。***
