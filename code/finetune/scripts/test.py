import os
import importlib.util
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.dataset import ECGDataset
from modules.evaluator import test
from configs.config import Config
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(dataset_name):
    # seed = 3407
    # set_seed(seed)

    config = Config()

    print(f"\n测试数据集: {dataset_name}")
    device = torch.device(config.device)

    num_workers = 6
    
    # 根据配置选择使用预处理数据还是原始数据
    test_dataset = ECGDataset(
        config.dataset_root, dataset_name, mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # 根据数据集名称查找对应的weights.pth路径
    base_path = os.path.join(config.checkpoint_dir, "model_snapshots", "vim1d")
    checkpoint_path = os.path.join(base_path, dataset_name, "weights.pth")

    if not os.path.exists(checkpoint_path):
        print(f"警告: 未找到权重文件: {checkpoint_path}")
        return

    print(f"加载权重文件: {checkpoint_path}")

    model_path = os.path.join(os.path.dirname(checkpoint_path), "model.py")
    spec = importlib.util.spec_from_file_location("ECGModel", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    ECGModel = getattr(model_module, "ECGModel")

    model = ECGModel(
        num_classes=test_dataset.num_classes
    )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    acc, auc, sens, spec, f1, mcc = test(model, test_loader, device)

    print("\n" + "=" * 60)
    print(f"{'Test Results for ' + dataset_name:^60}")
    print("=" * 60)
    print(f"Accuracy:    {acc:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Sensitivity:    {sens:.4f}")
    print(f"Specificity:    {spec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"MCC:    {mcc:.4f}")
    print("=" * 60 + "\n")

    results_dir = os.path.dirname(checkpoint_path)
    results_path = os.path.join(results_dir, "test_metrics.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"AUC:       {auc:.4f}\n")
        f.write(f"Sensitivity:    {sens:.4f}\n")
        f.write(f"Specificity:    {spec:.4f}\n")
        f.write(f"F1 Score:    {f1:.4f}\n")
        f.write(f"MCC:    {mcc:.4f}\n")
    print(f"测试指标已保存到: {results_path}")


if __name__ == "__main__":
    dataset_names = [
        # 'ptbxl-form',
        # 'ptbxl-rhythm',
        'ptbxl-super',
        # 'ptbxl-sub',
        # 'chapman',
        # 'icbeb',
    ]
    for dataset_name in dataset_names:
        print("=" * 60)
        print(f"Testing {dataset_name}...")
        print("=" * 60)
        main(dataset_name)
        print("=" * 80)
