import torch
from pathlib import Path


class Config:
    def __init__(self):
        current_file_path = Path(__file__).resolve()
        # 项目根目录：third_paper_code/
        project_root = current_file_path.parents[3]
        # 数据集目录移动到上一级：../datasets/
        self.data_prefix_root = str(project_root.parent / "datasets")
        # 检查点目录：third_paper_code/checkpoints_pretrain/
        self.checkpoint_dir = str(project_root / "checkpoints_pretrain")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_distributed = True  # 启用DDP多卡训练
        self.dist_backend = "nccl"

        ### pretrain相关配置

        # 默认参数
        self.batch_size = 64
        self.lr_rate = 1e-4
        self.weight_decay = 1e-5
        self.enable_swanlab = True
        self.swanlab_project = "ecg-finetune"
        self.temperature = 0.07
        self.seed = 42
