import torch
from pathlib import Path


class Config:
    def __init__(self):
        current_file_path = Path(__file__).resolve()
        # 数据集根目录：指向 /home/jasonwei/projects/datasets
        self.data_prefix_root = str(current_file_path.parents[3].parent / "datasets")
        # 检查点目录：当前pretrain目录下的checkpoints_pretrain
        self.checkpoint_dir = str(current_file_path.parents[1] / "checkpoints_pretrain")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_distributed = False
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
