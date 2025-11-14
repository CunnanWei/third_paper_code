# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于Vision Mamba (ViM)模型的ECG（心电图）信号分类项目，用于微调预训练模型进行下游ECG分类任务。项目支持多个ECG数据集，包括PTB-XL的多个子集、Chapman和ICBEB数据集。

## 常用命令

### 训练模型
```bash
# 训练单个数据集
python scripts/train.py

# 训练多个数据集（在scripts/train.py中修改dataset_name列表）
# 支持的数据集: ptbxl-form, ptbxl-rhythm, ptbxl-super, ptbxl-sub, chapman, icbeb
```

### 测试模型
```bash
# 测试所有数据集
python scripts/test.py

# 单独测试特定数据集（修改scripts/test.py中的dataset_names列表）
```

### 数据预处理
```bash
# 预处理原始数据（如果需要从原始数据重新预处理）
python scripts/preprocess_data.py --dataset [dataset_name] --mode [train/val/test]
```

## 项目架构

### 核心模块结构

1. **configs/** - 配置文件
   - `config.py`: 统一配置管理，包含数据路径、超参数等

2. **modules/** - 核心功能模块
   - `model.py`: ECGModel主模型类，基于ViM-small架构
   - `trainer.py`: Trainer训练器，支持混合精度训练、学习率调度
   - `evaluator.py`: 模型评估，计算各项指标（AUC、准确率等）
   - `dataset.py`: ECGDataset数据加载类，支持预处理和原始数据
   - `metrics.py`: 评估指标计算函数

3. **modules/layers/** - 模型层组件
   - `vim.py`: Vision Mamba核心实现
   - `timesformer_pytorch/`: TimesFormer相关实现
   - 其他网络层：resnet1d、vit1d、inception1d等

4. **scripts/** - 执行脚本
   - `train.py`: 训练脚本
   - `test.py`: 测试脚本
   - `preprocess_data.py`: 数据预处理脚本

5. **checkpoints_finetune/**（与 `code/` 同级）- 模型检查点
   - `model_snapshots/vim1d/`: 按数据集组织的模型权重

### 数据流程

1. **数据路径配置**: 通过`configs/config.py`设置数据集根目录
   - 预处理数据: `{project_root}/datasets/preprocessed-1d/{dataset_name}/`
   - 原始数据: 支持PTB-XL、Chapman、ICBEB格式

2. **训练数据流**:
   - ECGDataset → DataLoader → Trainer → 模型保存至checkpoints

3. **测试数据流**:
   - 加载checkpoints → Evaluator → 计算指标并保存结果

### 关键配置

- **默认模型**: ViM-small (Vision Mamba)
- **输入格式**: (batch_size, num_leads=12, seq_len=5000)
- **支持数据集**: ptbxl-super/form/rhythm/sub, chapman, icbeb
- **优化器**: AdamW (lr=1e-4, weight_decay=0.05)
- **调度器**: Linear预热 + Cosine退火
- **训练轮数**: 50 epochs (早停patience=5)

### 模型特点

- 支持混合精度训练提升训练效率
- 使用梯度裁剪稳定训练过程
- 实现早停机制防止过拟合
- 自动保存最佳验证AUC模型权重
- 支持多GPU并行数据加载

## 开发注意事项

- 修改数据集路径请检查`configs/config.py`中的路径配置
- 新增数据集需要在`modules/dataset.py`中添加相应的数据路径配置
- 模型权重会自动保存到项目根目录的`checkpoints_finetune/model_snapshots/vim1d/`目录
- 测试结果会保存到各数据集目录下的`test_metrics.txt`文件
