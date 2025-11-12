import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "layers" / "Vim" / "mamba-1p1p1"))
from layers.MedMamba import medmamba_t, medmamba_s, medmamba_b

class ECGModel(nn.Module):

    def __init__(
        self,
        num_classes,
        num_leads=12,):
        super().__init__()


        self.model = medmamba_t(
            patch_size=5,
            d_state=32,
            in_chans=num_leads,
            num_classes=num_classes,
        )

    def forward(self, x):

        logits = self.model(x)

        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试ECGModel
    print("测试ECGModel...")
    batch_size = 2
    num_leads = 12  # 12导联
    seq_len = 5000
    num_classes = 5

    # 创建测试数据 [batch, leads, seq_len]
    ecg_data = torch.randn(batch_size, num_leads, seq_len)
    print(f"ECG输入形状: {ecg_data.shape}")

    model = ECGModel(
        num_classes=num_classes,
        num_leads=num_leads,
        # seq_len=seq_len,
    ).to(device)

    ecg_data = ecg_data.to(device)
    with torch.no_grad():
        output = model(ecg_data)
        print(f"模型输出形状: {output.shape}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n测试完成！")
