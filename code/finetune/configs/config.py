import torch
from pathlib import Path


class Config:
    def __init__(self):
        current_file_path = Path(__file__).resolve()
        self.finetune_root = current_file_path.parents[1]
        project_root = self.finetune_root.parents[1]
        # 数据集与预处理结果放在项目上一级目录：../datasets/
        self.dataset_root = str(project_root.parent / "datasets" / "preprocessed-1d")
        self.checkpoint_dir = str(project_root / "checkpoints_finetune")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dataset_name = "ptbxl-super"
        self.batch_size = 64
        self.lr_rate = 1e-4  
        self.weight_decay = 1e-4
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.model_name = "resnet18_linear"
        self.linear_eval = True
        self.pretrain_weights = str(project_root / "checkpoints_pretrain" / "resnet18" / "weights.pth")
