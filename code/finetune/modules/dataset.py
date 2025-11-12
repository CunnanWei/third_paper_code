import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import wfdb
from scipy.io import loadmat
from scipy import signal
from torch_ecg._preprocessors import PreprocManager
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""
In this code:
PTB-XL has four subset: superclass, subclass, form, rhythm
ICBEB is CPSC2018 dataset mentioned in the original paper
Chapman is the CSN dataset from the original paper
"""


class ECGDataset(Dataset):
    """读取预处理后的ECG数据集"""
    def __init__(self, data_root, dataset_name, mode="train", ratio=None):
        """
        Args:
            preprocessed_root: 预处理数据根目录，如 "/home/jasonwei/projects/datasets/preprocessed-1d"
            dataset_name: 数据集名称，如 "ptbxl-super"
            mode: "train", "val", "test"
            ratio: 仅用于训练集，采样百分比 (1-100)
        """
        self.dataset_name = dataset_name
        self.mode = mode
        
        # 读取索引CSV
        dataset_dir = os.path.join(data_root, dataset_name)
        index_csv = os.path.join(dataset_dir, f"{mode}_index.csv")
        
        csv_data = pd.read_csv(index_csv)
        
        # 如果是训练集且指定了ratio，进行采样
        if mode == "train" and ratio is not None:
            total_rows = len(csv_data)
            n_samples = max(128, int(total_rows * ratio / 100))
            csv_data = csv_data.iloc[:n_samples]
        
        self.csv_data = csv_data
        
        # 数据目录
        self.data_dir = os.path.join(dataset_dir, mode)
        
        # 标签列（排除前4列：sample_id, data_file, original_ecg_path, original_csv_row）
        self.labels_name = list(self.csv_data.columns[4:])
        self.num_classes = len(self.labels_name)
        
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        
        data_file = os.path.join(self.data_dir, self.csv_data.iloc[idx]["data_file"])
        loaded = np.load(data_file, allow_pickle=True)

        data_dict = loaded.item()
        ecg = data_dict["data"]
        label = data_dict["label"]

        
        ecg = torch.from_numpy(ecg).float()
        label = torch.from_numpy(label).float()
        return ecg, label















# DATASET_PATHS = {
#     "ptbxl-super": {
#         "data_subdir": "ptbxl",
#         "splits": {
#             "train": "data_split/ptbxl/super_class/ptbxl_super_class_train.csv",
#             "val": "data_split/ptbxl/super_class/ptbxl_super_class_val.csv",
#             "test": "data_split/ptbxl/super_class/ptbxl_super_class_test.csv",
#         },
#     },
#     "ptbxl-sub": {
#         "data_subdir": "ptbxl",
#         "splits": {
#             "train": "data_split/ptbxl/sub_class/ptbxl_sub_class_train.csv",
#             "val": "data_split/ptbxl/sub_class/ptbxl_sub_class_val.csv",
#             "test": "data_split/ptbxl/sub_class/ptbxl_sub_class_test.csv",
#         },
#     },
#     "ptbxl-form": {
#         "data_subdir": "ptbxl",
#         "splits": {
#             "train": "data_split/ptbxl/form/ptbxl_form_train.csv",
#             "val": "data_split/ptbxl/form/ptbxl_form_val.csv",
#             "test": "data_split/ptbxl/form/ptbxl_form_test.csv",
#         },
#     },
#     "ptbxl-rhythm": {
#         "data_subdir": "ptbxl",
#         "splits": {
#             "train": "data_split/ptbxl/rhythm/ptbxl_rhythm_train.csv",
#             "val": "data_split/ptbxl/rhythm/ptbxl_rhythm_val.csv",
#             "test": "data_split/ptbxl/rhythm/ptbxl_rhythm_test.csv",
#         },
#     },
#     "icbeb": {
#         "data_subdir": "icbeb",
#         "splits": {
#             "train": "data_split/icbeb/icbeb_train.csv",
#             "val": "data_split/icbeb/icbeb_val.csv",
#             "test": "data_split/icbeb/icbeb_test.csv",
#         },
#     },
#     "chapman": {
#         "data_subdir": ".",
#         "splits": {
#             "train": "data_split/chapman/chapman_train.csv",
#             "val": "data_split/chapman/chapman_val.csv",
#             "test": "data_split/chapman/chapman_test.csv",
#         },
#     },
# }


# class ECGDataset(Dataset):
#     def __init__(
#         self, dataset_root, dataset_name, mode="train", ratio=None
#     ):

#         self.dataset_name = dataset_name
#         self.mode = mode


#         spec = DATASET_PATHS[dataset_name]
#         data_subdir = spec["data_subdir"]
#         self.data_path = os.path.join(
#             dataset_root, data_subdir) if data_subdir not in ("", ".") else dataset_root

#         csv_path = os.path.join(dataset_root, spec["splits"][mode])

#         if mode == "train" and ratio:
#             # 先读取行数，避免重复读取
#             total_rows = sum(1 for _ in open(csv_path)) - 1
#             csv_data = pd.read_csv(
#                 csv_path, nrows=max(128, int(total_rows * ratio / 100))
#             )
#         else:
#             csv_data = pd.read_csv(csv_path)

#         self.csv_data = csv_data

#         if self.dataset_name in [
#             "ptbxl-super",
#             "ptbxl-sub",
#             "ptbxl-form",
#             "ptbxl-rhythm",
#         ]:
#             self.labels_name = list(csv_data.columns[6:])
#             self.num_classes = len(self.labels_name)

#             self.ecg_path = csv_data["filename_hr"]
#             self.labels = csv_data.iloc[:, 6:].values

#         elif self.dataset_name == "icbeb":
#             self.labels_name = list(csv_data.columns[7:])
#             self.num_classes = len(self.labels_name)

#             self.ecg_path = csv_data["filename"].astype(str)
#             self.labels = csv_data.iloc[:, 7:].values

#         elif self.dataset_name == "chapman":
#             self.labels_name = list(csv_data.columns[3:])
#             self.num_classes = len(self.labels_name)

#             self.ecg_path = csv_data["ecg_path"].astype(str)
#             self.labels = csv_data.iloc[:, 3:].values

#         else:
#             raise ValueError(
#                 "dataset_type should be either 'ptbxl' or 'icbeb' or 'chapman"
#             )

#         self.preproc_config = {
#             "random": False,
#             "baseline_remove": {"window1": 0.2, "window2": 0.6},
#             "bandpass": {
#                 "lowcut": 0.5,
#                 "highcut": 45,
#                 "filter_type": "butter",
#                 "filter_order": 4,
#             },
#             "normalize": {"method": "z-score"},
#         }
#         self.ppm = PreprocManager.from_config(self.preproc_config)

#     def __len__(self):
#         return len(self.csv_data)

#     def __getitem__(self, idx):
#         if self.dataset_name.startswith("ptbxl"):
#             ecg_path = os.path.join(self.data_path, self.ecg_path[idx])
#             # the wfdb format file include ecg and other meta data
#             # the first element is the ecg data
#             ecg = wfdb.rdsamp(ecg_path)[0]
#             # the raw ecg shape is (5000, 12)
#             # transform to (12, 5000)
#             ecg = ecg.T

#             ecg = ecg[:, :5000]

#             ecg = torch.from_numpy(ecg).float()
#             target = self.labels[idx]
#             target = torch.from_numpy(target).float()

#         elif self.dataset_name == "icbeb":
#             # 对于 icbeb 数据集，处理文件名格式
#             ecg_id = self.ecg_path[idx]

#             ecg_path = os.path.join(self.data_path, ecg_id)

#             # icbeb has dat file, which is the raw ecg data
#             ecg = wfdb.rdsamp(ecg_path)
#             # the raw ecg shape is (n, 12), n is different for each sample
#             # transform to (12, n)
#             ecg = ecg[0].T
#             # icbeb has different length of ecg, so we need to preprocess it to the same length
#             # we only keep the first 2500 points as METS did
#             ecg = ecg[:, :2500]

#             # padding to 5000 to match the pre-trained data length
#             ecg = np.pad(ecg, ((0, 0), (0, 2500)),
#                          "constant", constant_values=0)
#             ecg = ecg[:, :5000]

#             ecg = torch.from_numpy(ecg).float()
#             target = self.labels[idx]
#             target = torch.from_numpy(target).float()

#         elif self.dataset_name == "chapman":
#             # chapman ecg_path has / at the start, so we need to remove it
#             ecg_path = os.path.join(self.data_path, self.ecg_path[idx][1:])
#             # raw data is (12, 5000), do not need to transform
#             ecg = loadmat(ecg_path)["val"]
#             ecg = ecg.astype(np.float32)

#             ecg = ecg[:, :5000]

#             ecg = torch.from_numpy(ecg).float()
#             target = self.labels[idx]
#             target = torch.from_numpy(target).float()

#         # switch AVL and AVF
#         # In MIMIC-ECG, the lead order is I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6
#         # In downstream datasets, the lead order is I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
#         ecg[[4, 5]] = ecg[[5, 4]]

#         ecg, _ = self.ppm(ecg, 500)

#         return ecg, target
