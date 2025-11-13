import torch
from pathlib import Path


class Config:
    def __init__(self):
        current_file_path = Path(__file__).resolve()
        # 项目根目录：third_paper_code/
        project_root = current_file_path.parents[3]
        # 数据集目录：third_paper_code/datasets/
        self.data_prefix_root = str(project_root / "datasets")
        # 检查点目录：third_paper_code/code/pretrain/checkpoints_pretrain/
        self.checkpoint_dir = str(current_file_path.parents[1] / "checkpoints_pretrain")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_distributed = True  # 启用DDP多卡训练
        self.dist_backend = "nccl"

        ### pretrain相关配置

        # 默认参数
        self.batch_size = 128
        self.lr_rate = 1e-4
        self.weight_decay = 1e-5
        self.enable_wandb = False
        self.wandb_project = "ecg-pretrain"
        self.text_model = "ncbi/MedCPT-Query-Encoder"
        self.text_max_length = 256
        self.max_logit_scale = 4.605170185988092
