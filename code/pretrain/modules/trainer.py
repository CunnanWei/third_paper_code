import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.cuda.amp import autocast


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        scheduler=None,
        is_main_process=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.is_main_process = is_main_process
        self.model_ref = self.model.module if hasattr(self.model, "module") else self.model

    def _step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def train(self, dataloader, scaler=None):
        self.model.train()
        stats = {
            "total_loss": 0.0,
            "processed_samples": 0,
        }

        pbar = tqdm(
            dataloader,
            desc="Train",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour="blue",
            disable=not self.is_main_process,
        )

        for batch in pbar:

            ecg, input_ids = batch
            ecg = ecg.to(self.device, non_blocking=True)
            input_ids = input_ids.to(self.device, non_blocking=True)
            model_inputs = (ecg, input_ids)

            # 前向传播
            self.optimizer.zero_grad()

            # 使用混合精度训练
            if scaler is not None:
                with autocast():
                    loss = self.model(*model_inputs)

                # 反向传播
                scaler.scale(loss).backward()
                # 梯度裁剪 - 稳定训练过程
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss = self.model(*model_inputs)

                # 反向传播
                loss.backward()
                # 梯度裁剪 - 稳定训练过程
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self._step_scheduler()

            # 更新统计信息
            batch_size = ecg.size(0)
            stats["total_loss"] += loss.detach().item() * batch_size
            stats["processed_samples"] += batch_size

            # 更新进度条
            if self.is_main_process:
                current_temp = torch.exp(
                    self.model_ref.clip.temperature.detach()
                ).item()
                pbar.set_postfix({
                    "loss": f"{stats['total_loss']/stats['processed_samples']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    "temp": f"{current_temp:.3f}",
                })

        avg_loss = stats["total_loss"] / stats["processed_samples"]
        return avg_loss

    def validate(self, dataloader):
        """验证模型，返回损失和对比准确率"""
        self.model.eval()
        total_loss = 0.0
        processed_samples = 0

        # 对比学习准确率统计
        total_t2i_top1 = 0.0  # text-to-image top1
        total_t2i_top5 = 0.0
        total_i2t_top1 = 0.0  # image-to-text top1
        total_i2t_top5 = 0.0

        with torch.no_grad():
            pbar = tqdm(
                dataloader,
                desc="Valid",
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="green",
                disable=not self.is_main_process,
            )

            for batch in pbar:
                ecg, input_ids = batch
                ecg = ecg.to(self.device, non_blocking=True)
                input_ids = input_ids.to(self.device, non_blocking=True)

                # 获取相似度矩阵用于计算准确率
                text_latents, image_latents = self.model_ref.clip(
                    input_ids,
                    ecg,
                    return_latents=True,
                )
                sim = text_latents @ image_latents.t()
                sim = sim * torch.exp(self.model_ref.clip.temperature)

                # 手动计算loss
                batch_size = sim.shape[0]
                labels = torch.arange(batch_size, device=sim.device)
                loss = (torch.nn.functional.cross_entropy(sim, labels) +
                        torch.nn.functional.cross_entropy(sim.t(), labels)) / 2

                # 计算对比学习准确率
                # Text-to-Image: 每个text找最相似的image
                t2i_pred = sim.argmax(dim=1)
                t2i_top1 = (t2i_pred == labels).float().sum().item()

                # Top-5
                t2i_top5_pred = sim.topk(min(5, batch_size), dim=1)[1]
                t2i_top5 = sum([labels[i].item() in t2i_top5_pred[i].tolist()
                               for i in range(batch_size)])

                # Image-to-Text: 每个image找最相似的text
                i2t_pred = sim.t().argmax(dim=1)
                i2t_top1 = (i2t_pred == labels).float().sum().item()

                # Top-5
                i2t_top5_pred = sim.t().topk(min(5, batch_size), dim=1)[1]
                i2t_top5 = sum([labels[i].item() in i2t_top5_pred[i].tolist()
                               for i in range(batch_size)])

                total_loss += loss.item() * batch_size
                total_t2i_top1 += t2i_top1
                total_t2i_top5 += t2i_top5
                total_i2t_top1 += i2t_top1
                total_i2t_top5 += i2t_top5
                processed_samples += batch_size

                # 更新进度条
                if self.is_main_process:
                    pbar.set_postfix({
                        "loss": f"{total_loss/processed_samples:.4f}",
                        "t2i_top1": f"{100*total_t2i_top1/processed_samples:.1f}%",
                        "i2t_top1": f"{100*total_i2t_top1/processed_samples:.1f}%",
                    })

        if processed_samples == 0:
            return {"loss": 0.0}

        # 在DDP模式下，聚合所有进程的总和，然后计算全局平均
        if dist.is_initialized():
            # 聚合分子和分母
            stats_tensor = torch.tensor([
                total_loss,
                total_t2i_top1,
                total_t2i_top5,
                total_i2t_top1,
                total_i2t_top5,
                processed_samples
            ], dtype=torch.float32, device=self.device)
            
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            
            # 解包聚合后的统计数据
            total_loss = stats_tensor[0].item()
            total_t2i_top1 = stats_tensor[1].item()
            total_t2i_top5 = stats_tensor[2].item()
            total_i2t_top1 = stats_tensor[3].item()
            total_i2t_top5 = stats_tensor[4].item()
            processed_samples = stats_tensor[5].item()

        # 计算全局指标
        metrics = {
            "loss": total_loss / processed_samples,
            "text_to_image_top1": total_t2i_top1 / processed_samples,
            "text_to_image_top5": total_t2i_top5 / processed_samples,
            "image_to_text_top1": total_i2t_top1 / processed_samples,
            "image_to_text_top5": total_i2t_top5 / processed_samples,
            "mean_recall": (total_t2i_top1 + total_i2t_top1) / (2 * processed_samples),
        }

        return metrics
