import torch
import numpy as np
import pandas as pd
import os
import wfdb
from scipy.io import loadmat
from torch_ecg._preprocessors import PreprocManager
import warnings
from tqdm import tqdm
import argparse
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 导入与dataset.py一致的数据集配置
import sys
# 当前文件: code/finetune/scripts/preprocess_data.py
# 需要添加: code/finetune (向上一级)
finetune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if finetune_root not in sys.path:
    sys.path.append(finetune_root)
from modules.dataset import DATASET_PATHS


class ECGPreprocessor:
    def __init__(self):
        self.signal_length = 5000

        # 信号预处理配置
        self.preproc_config = {
            "random": False,
            "baseline_remove": {"window1": 0.2, "window2": 0.6},
            "bandpass": {
                "lowcut": 0.5,
                "highcut": 45,
                "filter_type": "butter",
                "filter_order": 4,
            },
            "normalize": {"method": "z-score"},
        }
        self.ppm = PreprocManager.from_config(self.preproc_config)

    def load_ecg_signal(self, data_path, ecg_path, dataset_name):
        """加载ECG信号数据，与dataset.py保持完全一致"""
        if dataset_name.startswith("ptbxl"):
            ecg_file_path = os.path.join(data_path, ecg_path)
            # the wfdb format file include ecg and other meta data
            # the first element is the ecg data
            ecg = wfdb.rdsamp(ecg_file_path)[0]
            # the raw ecg shape is (5000, 12)
            # transform to (12, 5000)
            ecg = ecg.T
            ecg = ecg[:, :5000]

        elif dataset_name == "icbeb":
            # 对于 icbeb 数据集，处理文件名格式
            ecg_file_path = os.path.join(data_path, ecg_path)
            # icbeb has dat file, which is the raw ecg data
            ecg = wfdb.rdsamp(ecg_file_path)
            # the raw ecg shape is (n, 12), n is different for each sample
            # transform to (12, n)
            ecg = ecg[0].T
            # icbeb has different length of ecg, so we need to preprocess it to the same length
            # we only keep the first 2500 points as METS did
            ecg = ecg[:, :2500]
            # padding to 5000 to match the pre-trained data length
            ecg = np.pad(ecg, ((0, 0), (0, 2500)),
                         "constant", constant_values=0)
            ecg = ecg[:, :5000]

        elif dataset_name == "chapman":
            # chapman ecg_path has / at the start, so we need to remove it
            ecg_file_path = os.path.join(data_path, ecg_path[1:])
            # raw data is (12, 5000), do not need to transform
            ecg = loadmat(ecg_file_path)["val"]
            ecg = ecg.astype(np.float32)
            ecg = ecg[:, :5000]

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return ecg

    def preprocess_sample(self, data_path, ecg_path, label, dataset_name):
        """预处理单个样本：返回与当前 Dataset 一致的 1D 波形 (12, 5000) 和标签"""
        # 1. 加载ECG信号
        ecg = self.load_ecg_signal(data_path, ecg_path, dataset_name)

        # 2. 交换 aVL 与 aVF 导联以匹配下游数据导联顺序
        ecg[[4, 5]] = ecg[[5, 4]]

        # 3. 使用与 Dataset 相同的预处理流程（去基线、带通滤波、标准化）
        ecg = torch.from_numpy(ecg).float()
        ecg, _ = self.ppm(ecg, 500)

        # 4. 返回 numpy 以便保存
        return ecg, label


def get_dataset_config(dataset_name, dataset_root):
    """获取数据集配置，与dataset.py保持完全一致"""
    spec = DATASET_PATHS[dataset_name]
    data_subdir = spec["data_subdir"]
    data_path = os.path.join(
        dataset_root, data_subdir) if data_subdir not in ("", ".") else dataset_root

    # 构建CSV路径
    csv_paths = {}
    for split in ["train", "val", "test"]:
        csv_paths[split] = os.path.join(dataset_root, spec["splits"][split])

    # 根据数据集类型确定标签起始列和路径列
    if dataset_name in ["ptbxl-super", "ptbxl-sub", "ptbxl-form", "ptbxl-rhythm"]:
        label_start_col = 6
        path_col = "filename_hr"
    elif dataset_name == "icbeb":
        label_start_col = 7
        path_col = "filename"
    elif dataset_name == "chapman":
        label_start_col = 3
        path_col = "ecg_path"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return data_path, csv_paths, label_start_col, path_col


def preprocess_dataset(dataset_name, dataset_root, output_root):
    """预处理整个数据集"""
    print(f"开始预处理数据集: {dataset_name}")

    # 获取数据集配置
    data_path, csv_paths, label_start_col, path_col = get_dataset_config(
        dataset_name, dataset_root
    )

    # 创建预处理器
    preprocessor = ECGPreprocessor()

    # 创建输出目录
    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # 获取标签名称（用于生成索引CSV）
    original_csv = pd.read_csv(csv_paths["train"])  # 读取训练集获取标签信息

    if dataset_name in ["ptbxl-super", "ptbxl-sub", "ptbxl-form", "ptbxl-rhythm"]:
        labels_name = list(original_csv.columns[6:])
    elif dataset_name == "icbeb":
        labels_name = list(original_csv.columns[7:])
    elif dataset_name == "chapman":
        labels_name = list(original_csv.columns[3:])

    total_samples = 0
    for split in ["train", "val", "test"]:
        print(f"处理 {split} 数据...")

        # 读取CSV文件
        csv_file = csv_paths[split]
        if not os.path.exists(csv_file):
            print(f"警告: CSV文件不存在: {csv_file}")
            continue

        csv_data = pd.read_csv(csv_file)

        # 创建输出子目录
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        # 准备索引数据
        index_data = {
            "sample_id": [],
            "data_file": [],
            "original_ecg_path": [],
            "original_csv_row": []  # 记录在原始CSV中的行索引
        }
        # 添加标签列
        for label_name in labels_name:
            index_data[label_name] = []

        # 处理每个样本
        split_samples = 0
        for csv_idx, row in tqdm(csv_data.iterrows(), total=len(csv_data), desc=f"{split}"):
            try:
                # 获取样本信息，与dataset.py保持一致
                if dataset_name == "icbeb":
                    ecg_path = str(row[path_col])
                elif dataset_name == "chapman":
                    ecg_path = str(row[path_col])
                else:
                    ecg_path = row[path_col]

                labels = row.iloc[label_start_col:].values.astype(np.float32)

                # 生成样本ID（使用行索引）
                sample_id = f"{csv_idx:06d}"

                # 预处理样本
                ecg_data, label_data = preprocessor.preprocess_sample(
                    data_path, ecg_path, labels, dataset_name
                )

                # 保存样本
                data_filename = f"{sample_id}.npy"
                output_file = os.path.join(split_output_dir, data_filename)
                np.save(
                    output_file,
                    {
                        "data": ecg_data,
                        "label": label_data,
                        "sample_id": sample_id,
                        "ecg_path": ecg_path,
                    },
                )

                # 记录到索引数据
                index_data["sample_id"].append(sample_id)
                index_data["data_file"].append(data_filename)
                index_data["original_ecg_path"].append(ecg_path)
                index_data["original_csv_row"].append(csv_idx)

                # 添加标签数据
                for i, label_name in enumerate(labels_name):
                    index_data[label_name].append(labels[i])

                split_samples += 1

            except Exception as e:
                print(f"处理样本 {csv_idx} 时出错: {e}")
                continue

        # 保存索引CSV文件
        if split_samples > 0:
            index_df = pd.DataFrame(index_data)
            index_csv_path = os.path.join(output_dir, f"{split}_index.csv")
            index_df.to_csv(index_csv_path, index=False)
            print(f"已保存索引文件: {index_csv_path}")

        print(f"{split} 完成，处理了 {split_samples} 个样本")
        total_samples += split_samples

    print(f"数据集 {dataset_name} 预处理完成，总共处理了 {total_samples} 个样本")


def main():
    parser = argparse.ArgumentParser(description="预处理ECG数据集")

    # 与 config 保持一致的默认路径（基于项目目录推断）
    # 当前文件位于 code/finetune/scripts/preprocess_data.py
    # 数据集实际在 /home/jasonwei/projects/datasets/finetune/
    current_file_path = os.path.abspath(__file__)
    code_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))  # code/finetune -> code
    projects_dir = os.path.dirname(os.path.dirname(code_dir))  # projects
    default_dataset_root = os.path.join(projects_dir, "datasets", "finetune")
    default_output_root = os.path.join(projects_dir, "datasets", "preprocessed-1d")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "ptbxl-super",
            "ptbxl-sub",
            "ptbxl-form",
            "ptbxl-rhythm",
            "icbeb",
            "chapman",
            "all",
        ],
        default="all",
        help="要处理的数据集",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=default_dataset_root,
        help="数据集根目录",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=default_output_root,
        help="预处理数据输出根目录",
    )

    args = parser.parse_args()

    # 创建输出根目录
    os.makedirs(args.output_root, exist_ok=True)

    if args.dataset == "all":
        datasets = [
            "ptbxl-super",
            "ptbxl-sub",
            "ptbxl-form",
            "ptbxl-rhythm",
            "icbeb",
            "chapman",
        ]
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        try:
            preprocess_dataset(dataset, args.dataset_root, args.output_root)
        except Exception as e:
            print(f"处理数据集 {dataset} 时出错: {e}")
            continue

    print("所有数据集预处理完成！")


if __name__ == "__main__":
    main()
