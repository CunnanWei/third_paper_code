import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.cuda.amp import GradScaler
import time
import random
import numpy as np
import warnings
import sys
from pathlib import Path
import swanlab
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import Config
from modules.dataset import ECGDataset_pretrain
from modules.model import ECGPretrainModel
from modules.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_distributed(config):
    use_distributed = config.enable_distributed and torch.cuda.device_count() > 1
    if not use_distributed:
        return False, 0, 0
    if not dist.is_initialized():
        dist.init_process_group(backend=config.dist_backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_rank()


def _run_training_with_config(config=None):
    if config is None:
        config = Config()
    use_distributed, local_rank, rank = init_distributed(config)
    is_main_process = (not use_distributed) or rank == 0

    if is_main_process:
        print("\n" + "=" * 60)
        print(f"{'Model Training Pipeline':^60}")
        print("=" * 60)
    seed = 42
    set_seed(seed)
    save_prefix = f"{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}"
    swanlab_run = None
    if config.enable_swanlab and is_main_process:
        swanlab_run = swanlab.init(
            project=config.swanlab_project,
            experiment_name=save_prefix,
            config={
                "batch_size": config.batch_size,
                "learning_rate": config.lr_rate,
                "weight_decay": config.weight_decay,
            },
        )

    if use_distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(config.device)

    if is_main_process:
        print("\nCONFIGURATION")
        print(f"  {'Batch size':18}: {config.batch_size}")
        print(f"  {'Learning rate':18}: {config.lr_rate}")
        print(f"  {'Weight Decay':18}: {config.weight_decay}")
        print(f"  {'Device':18}: {device}")
        print("\nPREPARING DATA...")

    train_dataset = ECGDataset_pretrain(config, mode="train")
    val_dataset = ECGDataset_pretrain(config, mode="val")

    num_workers = 12

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
    ) if use_distributed else None

    # 为验证集也添加DistributedSampler，确保DDP模式下每个进程验证不同的数据子集
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False,  # 验证时不需要shuffle
    ) if use_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 验证集不需要shuffle
        sampler=val_sampler,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=8,
        drop_last=True,  # DDP模式下保持各进程batch数量一致
    )

    if is_main_process:
        print(f"  {'Train samples':18}: {len(train_dataset)}")
        print(f"  {'Val samples':18}: {len(val_dataset)}")

    model = ECGPretrainModel(config).to(device)
    if use_distributed:
        # find_unused_parameters=True: 允许部分参数不参与梯度计算（如TextEncoder冻结层）
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr_rate, weight_decay=config.weight_decay
    )

    num_epochs = 100  # 大数据集下，每个epoch已经充分，30-50 epochs即可

    num_batches_per_epoch = len(train_loader)
    total_steps = max(1, num_epochs * num_batches_per_epoch)

    # 预热步数按总步数比例自适应，并限制在 [1, 4000] 内
    warmup_ratio = 0.03
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_cap = min(int(total_steps * 0.05), 4000)
    warmup_steps = max(1, min(warmup_steps, warmup_cap))
    if warmup_steps >= total_steps:
        warmup_steps = max(1, total_steps - 1)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_steps,
    )

    # 余弦部分按总剩余步数平均切成 2-3 个周期，使重启节奏随训练规模调整
    remaining_steps = max(1, total_steps - warmup_steps)
    cosine_cycles = 3 if num_epochs >= 60 else 2 if num_epochs >= 30 else 1
    steps_per_cycle = max(1, remaining_steps // cosine_cycles)

    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=steps_per_cycle,
        T_mult=1,
        eta_min=config.lr_rate * 0.001,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps - 1],
    )

    # 添加混合精度训练支持
    scaler = GradScaler() if torch.cuda.is_available() else None

    trainer = Trainer(
        model,
        optimizer,
        device,
        scheduler=scheduler,
        is_main_process=is_main_process,
    )

    # 提前定义模型保存目录，所有进程使用相同路径
    model_dir = os.path.join(
        config.checkpoint_dir,
        f"ckpt_{save_prefix}",
    )

    if is_main_process:
        print("\n" + "-" * 60)
        print(f"{'TRAINING STARTED':^60}")
        print("-" * 60)

    best_val_loss = float('inf')
    best_mean_recall = 0.0  # 对比学习准确率作为主要指标
    count_early_stop = 0
    patience = 10  # 大数据集每个epoch成本高，给予更多耐心

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # 为训练sampler设置epoch，确保每个epoch数据分布不同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # 为验证sampler也设置epoch，保证DDP模式下数据分片的确定性
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        if is_main_process:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = trainer.train(train_loader, scaler=scaler)
        val_metrics = trainer.validate(val_loader)


        # 在评估和保存模型前同步所有进程
        if use_distributed:
            dist.barrier()

        if is_main_process:
            print(
                "  Train loss: "
                f"{train_loss:.4f}"
            )
            print(
                "  Val loss: "
                f"{val_metrics['loss']:.4f}"
            )
            print(
                "  Text→ECG Top-1: "
                f"{val_metrics['text_to_image_top1']*100:.2f}% | "
                f"Top-5: {val_metrics['text_to_image_top5']*100:.2f}%"
            )
            print(
                "  ECG→Text Top-1: "
                f"{val_metrics['image_to_text_top1']*100:.2f}% | "
                f"Top-5: {val_metrics['image_to_text_top5']*100:.2f}%"
            )
            print(
                "  Mean Recall: "
                f"{val_metrics['mean_recall']*100:.2f}%"
            )
        if swanlab_run is not None and is_main_process:
            swanlab_run.log(
                {
                    # "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/text_to_ecg_top1": val_metrics["text_to_image_top1"],
                    "val/text_to_ecg_top5": val_metrics["text_to_image_top5"],
                    "val/ecg_to_text_top1": val_metrics["image_to_text_top1"],
                    "val/ecg_to_text_top5": val_metrics["image_to_text_top5"],
                    "val/mean_recall": val_metrics["mean_recall"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # 只在第一个epoch创建模型保存目录
        if epoch == 0 and is_main_process:
            os.makedirs(model_dir, exist_ok=True)
            print(f"Model snapshot directory created: {model_dir}")

        # 确保目录创建完成后再继续
        if use_distributed:
            dist.barrier()

        # 使用 Mean Recall 作为主要指标（对比学习准确率）
        current_mean_recall = val_metrics['mean_recall']
        if current_mean_recall > best_mean_recall:
            best_mean_recall = current_mean_recall
            best_val_loss = val_metrics['loss']
            if is_main_process:
                torch.save(trainer.model_ref.state_dict(), os.path.join(model_dir, "weights.pth"))
                print(
                    f"  Model saved! (Best Mean Recall: {best_mean_recall*100:.2f}%)")
            count_early_stop = 0
        else:
            count_early_stop += 1
            if is_main_process:
                print(f"  Early stopping count: {count_early_stop}/{patience}")
            if count_early_stop >= patience:
                if is_main_process:
                    print(f"  Early stopping triggered at epoch {epoch + 1}.")
                    print(f"  Best Mean Recall: {best_mean_recall*100:.2f}%")
                    print(f"  Best Val Loss: {best_val_loss:.4f}")
                break

    if swanlab_run is not None and is_main_process:
        swanlab_run.finish()

    if use_distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    """
    单机多卡训练入口
    使用torchrun启动: torchrun --nproc_per_node=<GPU数量> scripts/train.py
    单卡训练: python scripts/train.py
    """
    config = Config()
    _run_training_with_config(config)


if __name__ == "__main__":
    main()
