import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset_pretrain(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode

        if self.mode == "train":
            self.csv_path = "datasets/pretrain/train.csv"
        elif self.mode == "val":
            self.csv_path = "datasets/pretrain/val.csv"

        self.csv_data = pd.read_csv(self.csv_path, usecols=['path', 'report_tokenize_path'])

    def __len__(self):
        return len(self.csv_data)



    def __getitem__(self, idx):
        ecg_path = self.csv_data.iloc[idx]["path"]
        tokenize_output_path= self.csv_data.iloc[idx]["report_tokenize_path"]
        

        ecg = torch.from_numpy(np.load(ecg_path))
        tokenize_output = np.load(tokenize_output_path)
        # npz文件包含多个数组，通过键访问
        input_ids = torch.from_numpy(tokenize_output['input_ids'])

        return ecg, input_ids
