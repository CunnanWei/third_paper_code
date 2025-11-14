import torch
from pathlib import Path


class Config:
    def __init__(self):
        current_file_path = Path(__file__).resolve()
        self.finetune_root = current_file_path.parents[1]
        project_root = self.finetune_root.parents[1]
        self.dataset_root = str(project_root / "datasets" / "preprocessed-1d")
        self.checkpoint_dir = str(project_root / "checkpoints_finetune")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dataset_name = "ptbxl-super"
        self.batch_size = 64
        self.lr_rate = 1e-4  
        self.weight_decay = 1e-4
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.model_name = "medmamba_s"
