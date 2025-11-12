import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        warmup_scheduler=None,
        main_scheduler=None,
        warmup_steps=0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # 学习率调度器设置
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def _update_lr_scheduler(self):
        """更新学习率调度器"""
        if self.warmup_scheduler is None or self.main_scheduler is None:
            return

        self.current_step += 1
        # 确保在optimizer.step()之后调用scheduler.step()
        if self.current_step <= self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()

    def train(self, dataloader, scaler=None):
        self.model.train()
        stats = {
            "total_loss": 0.0,
            "processed_samples": 0,
        }

        pbar = tqdm(
            dataloader, desc="Train", bar_format="{l_bar}{bar:30}{r_bar}", colour="blue"
        )

        for batch in pbar:

            ecg, labels = batch
            ecg = ecg.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            # 前向传播
            self.optimizer.zero_grad()

            # 使用混合精度训练
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(ecg)
                    total_loss = self.criterion(logits, labels)

                # 反向传播
                scaler.scale(total_loss).backward()
                # 梯度裁剪 - 稳定训练过程
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                logits = self.model(ecg)
                total_loss = self.criterion(logits, labels)

                # 反向传播
                total_loss.backward()
                # 梯度裁剪 - 稳定训练过程
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self._update_lr_scheduler()

            # 更新统计信息
            batch_size = labels.size(0)
            stats["total_loss"] += total_loss.item() * batch_size
            stats["processed_samples"] += batch_size

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{stats['total_loss']/stats['processed_samples']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        return stats["total_loss"] / stats["processed_samples"]

    def validate(self, dataloader):
        """验证模型，返回损失和AUC"""
        self.model.eval()
        total_loss = 0.0
        processed_samples = 0
        labels_list, probs_list = [], []

        with torch.no_grad():
            pbar = tqdm(
                dataloader,
                desc="Valid",
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="green",
            )

            for batch in pbar:
                ecg, labels = batch
                ecg = ecg.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                logits = self.model(ecg)
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits)

                # 收集预测和标签用于AUC计算
                labels_list.append(labels.cpu().numpy())
                probs_list.append(probs.cpu().numpy())

                # 更新统计信息
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                processed_samples += batch_size

                # 更新进度条
                pbar.set_postfix({"loss": f"{total_loss/processed_samples:.4f}"})

        # 计算AUC
        labels_arr = np.vstack(labels_list)
        probs_arr = np.vstack(probs_list)
        auc = roc_auc_score(labels_arr, probs_arr)

        return total_loss / processed_samples, auc
