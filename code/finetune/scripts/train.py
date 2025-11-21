import warnings
import numpy as np
import random
import inspect
import shutil
import datetime
from datetime import timedelta
import time
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import torch
from torch.cuda.amp import GradScaler
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
from modules.trainer import Trainer
from modules.model import ECGModel
from modules.dataset import ECGDataset
from modules.evaluator import test
from configs.config import Config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_pretrained_backbone(model, pretrained_path):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith("ecg_encoder."):
            backbone_state[k.replace("ecg_encoder.", "model.")] = v
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"  Loaded pretrained backbone from: {pretrained_path}")
    print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

def main(dataset_name):
    print("\n" + "=" * 60)
    print(f"{'Model Training Pipeline':^60}")
    print("=" * 60)

    start_time = time.time()

    config = Config()
    seed = 42
    set_seed(seed)

    print("\nCONFIGURATION")
    print(f"  {'Dataset':18}: {dataset_name}")
    print(f"  {'Batch size':18}: {config.batch_size}")
    print(f"  {'Learning rate':18}: {config.lr_rate}")
    print(f"  {'Device':18}: {config.device}")

    device = torch.device(config.device)

    print("\nPREPARING DATA...")

    train_dataset = ECGDataset(
        config.dataset_root, dataset_name, mode="train"
    )
    val_dataset = ECGDataset(
        config.dataset_root, dataset_name, mode="val"
    )

    num_workers = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4,
    )

    print(f"  {'Train samples':18}: {len(train_dataset)}")
    print(f"  {'Val samples':18}: {len(val_dataset)}")

    model = ECGModel(
        num_classes=train_dataset.num_classes
    )

    if config.linear_eval:
        load_pretrained_backbone(model, config.pretrain_weights)
        for p in model.model.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(
            model.fc.parameters(), lr=config.lr_rate, weight_decay=config.weight_decay
        )
        print("  Linear evaluation: backbone frozen, training classifier head only")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr_rate, weight_decay=config.weight_decay
        )
        print("  Full finetuning: all parameters trainable")

    num_epochs = 50
    warmup_epochs = 5

    num_batches_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * num_batches_per_epoch

    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20 * num_batches_per_epoch,
        T_mult=2,
        eta_min=config.lr_rate * 0.01,
    )

    # 添加混合精度训练支持
    scaler = GradScaler() if torch.cuda.is_available() else None

    trainer = Trainer(
        model,
        optimizer,
        config.criterion,
        device,
        warmup_scheduler=warmup_scheduler,
        main_scheduler=main_scheduler,
        warmup_steps=warmup_steps,
    )

    print("\n" + "-" * 60)
    print(f"{'TRAINING STARTED':^60}")
    print("-" * 60)

    best_val_auc = 0.0
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    count_early_stop = 0
    patience = 20  # 匹配第2周期长度(40)的50%，给予warm restart充分探索空间  

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = trainer.train(train_loader, scaler=scaler)
        val_loss, val_auc = trainer.validate(val_loader)

        epoch_time = time.time() - epoch_start
        print(
            f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}  Val AUC: {val_auc:.4f}  Time: {str(timedelta(seconds=int(epoch_time)))}"
        )

        if epoch == 0:
            model_dir = os.path.join(
                config.checkpoint_dir,
                "model_snapshots",
                config.model_name,
                f"{dataset_name}",
            )
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy2(
                inspect.getfile(model.__class__), os.path.join(
                    model_dir, "model.py")
            )
            print(f"Model snapshot directory created: {model_dir}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(
                model_dir, "weights.pth"))
            print(f"  Weights updated!")
            count_early_stop = 0
        else:
            count_early_stop += 1
            print(f"  Early stopping count: {count_early_stop}/{patience}")
            if count_early_stop >= patience:
                print(f"  Early stopping triggered at epoch {epoch + 1}.")
                print(f"  Best Val AUC: {best_val_auc:.4f}")
                print(f"  Weights updated at: {model_dir}")
                break

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print("=" * 60 + "\n")

    # ========== TESTING PHASE ==========
    print("\n" + "=" * 60)
    print(f"{'TESTING PHASE':^60}")
    print("=" * 60)
    
    test_dataset = ECGDataset(
        config.dataset_root, dataset_name, mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"  {'Test samples':18}: {len(test_dataset)}")
    
    # 加载最佳模型权重
    best_weights_path = os.path.join(model_dir, "weights.pth")
    print(f"  Loading best weights from: {best_weights_path}")
    
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    
    # 运行测试
    print("\n  Running test...")
    acc, auc, sens, spec, f1, mcc = test(model, test_loader, device)
    
    # 打印测试结果
    print("\n" + "-" * 60)
    print(f"{'Test Results':^60}")
    print("-" * 60)
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  AUC:           {auc:.4f}")
    print(f"  Sensitivity:   {sens:.4f}")
    print(f"  Specificity:   {spec:.4f}")
    print(f"  F1 Score:      {f1:.4f}")
    print(f"  MCC:           {mcc:.4f}")
    print("-" * 60)
    
    # 保存测试指标到文件
    results_path = os.path.join(model_dir, "test_metrics.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Best Val AUC: {best_val_auc:.4f}\n\n")
        f.write(f"Test Metrics:\n")
        f.write(f"Accuracy:      {acc:.4f}\n")
        f.write(f"AUC:           {auc:.4f}\n")
        f.write(f"Sensitivity:   {sens:.4f}\n")
        f.write(f"Specificity:   {spec:.4f}\n")
        f.write(f"F1 Score:      {f1:.4f}\n")
        f.write(f"MCC:           {mcc:.4f}\n")
    
    print(f"\n  Test metrics saved to: {results_path}")
    print("=" * 60 + "\n")



if __name__ == "__main__":
    dataset_name = [
        'ptbxl-form',
        'ptbxl-rhythm',
        'ptbxl-super',
        'ptbxl-sub',
        'chapman',
        'icbeb',
    ]
    for name in dataset_name:
        print(f"Training {name}...")
        main(name)
