import torch.nn.functional as F
import torch.nn as nn
import torch
from pathlib import Path
from mamba_ssm import Mamba2

import sys
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "layers" / "x_clip"))
from layers.x_clip.x_clip import CLIP
from layers.text_encoder import TextEncoder
# from layers.timesformer_pytorch.timesformer_pytorch import TimeSformer
# from layers.resnet1d import ResNet18
from layers.MedMamba import medmamba_t, medmamba_s, medmamba_b

class ECGPretrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ecg_encoder = medmamba_t(
            patch_size=5,
            d_state=16,
            in_chans=12,
            num_classes=256,
        )

        self.text_encoder = TextEncoder()
        self.clip = CLIP(
            image_encoder=self.ecg_encoder,
            text_encoder=self.text_encoder,
            dim_image=256,
            dim_text=256,
            dim_latent=256,
            text_pad_id=self.text_encoder.tokenizer.pad_token_id,
        )

    def forward(self, ecg, input_ids):
        # CLIP内部会根据text_pad_id自动计算mask (text_mask = text != pad_token_id)
        # 对于标准tokenizer，这与预先生成的attention_mask等价
        # 
        # 注意：x_clip在eval模式下不支持return_loss=True (会触发assert)
        # 验证时的loss计算在trainer.py中通过return_latents=True手动完成
        if self.training:
            return self.clip(input_ids, ecg, return_loss=True)
        else:
            return self.clip(input_ids, ecg, return_loss=False)


if __name__ == "__main__":
    from configs.config import Config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecg = torch.randn(4, 12, 5000).to(device)
    cfg = Config()
    model = ECGPretrainModel(cfg).to(device)
