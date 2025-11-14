# Repository Guidelines

## 项目结构与模块组织
本仓库聚焦 ViM ECG 微调，核心代码位于 `code/finetune`。`modules/` 提供模型、数据、训练与评估组件；`modules/layers/` 收录 ViM、TimesFormer、ResNet1d 等子模块，可按需扩展；`configs/config.py` 统一管理数据根路径、优化器、scheduler 及 checkpoint 目录；`scripts/` 下的 `preprocess_data.py`、`train.py`、`test.py` 是唯一入口；训练产物保存到项目根目录的 `checkpoints_finetune/model_snapshots/vim1d/<dataset>/`。在切换数据集时，请同步更新 Config 的 `dataset_root` 与脚本内的 `dataset_name` 列表，避免路径和权重混淆。

## 构建、测试与开发命令
- `python scripts/preprocess_data.py --dataset ptbxl-super --mode train`：按需重建缓存，`mode` 取 train/val/test。
- `CUDA_VISIBLE_DEVICES=0 python scripts/train.py`：默认使用 Config 中的批大小、学习率及线性预热 + cosine 调度，日志会打印样本数与 AUC。
- `python scripts/test.py`：遍历 `dataset_names` 列表，对应 checkpoint 必须位于项目根目录的 `checkpoints_finetune/model_snapshots/vim1d/<dataset>/weights.pth`，结果会同时写入 `test_metrics.txt`。
- `PYTHONPATH=. python scripts/train.py`：若在外层调用，先确保 `PYTHONPATH` 指向仓库根目录。

## 编码风格与命名约定
采用 Python 3.10，统一四空格缩进，模块与函数使用 snake_case，类保持 PascalCase，配置常量使用大写。严禁引入动态 `await import(...)`、类型断言到 `Any` 或空的 `cast`；也不要为函数和变量添加多余的类型注解，本项目依赖推断与 Docstring。避免包裹额外的 try/except 或防御式判断，保持与当前实现一致的简洁控制流。新增层或指标时，请将逻辑拆分到 `modules/` 子文件，并暴露清晰的工厂函数。

## 测试准则
测试依托 `scripts/test.py` 与 `modules/evaluator.py::test`，默认 DataLoader num_workers=6。扩展模型后，请至少对一个 PTB-XL 数据集执行完整评估，并确认控制台 AUC 与 `test_metrics.txt` 一致。若引入新数据集，先运行预处理脚本，再更新 Config 的路径与 `ECGDataset` 调度逻辑。提交前应附上关键指标（Accuracy/AUC/F1/MCC）以及对应 checkpoint 路径，便于复现。

## 提交与 Pull Request 指南
当前子目录未发现 `.git` 元数据；若你在上层初始化 Git，请沿用简明动词短语，推荐 `<type>: <detail>`（如 `feat: extend trainer warmup scheduler`）。每个 PR 需要：1) 描述动机与关键改动；2) 附运行命令和核心指标；3) 链接相关议题或数据来源；4) 若更改 Config，说明默认设备与路径影响。评审者会重点验证训练脚本是否仍可在 `python scripts/train.py` 入口下运行，请在描述中明确依赖版本与最小 GPU 内存需求。

## 配置与安全提示
Config 中的 `checkpoint_dir` 默认为项目根目录下的 `checkpoints_finetune`，请确保对该路径有写权限，且不要把生产权重提交到公开仓库。敏感数据（患者 ID、原始 ECG CSV）应留在 `datasets/` 本地路径，脚本只引用匿名化张量文件。运行多 GPU 实验时，显式设置 `CUDA_VISIBLE_DEVICES`，并通过 `set_seed` 函数保持结果可比；上传日志前，检查是否含绝对路径或账号信息。
