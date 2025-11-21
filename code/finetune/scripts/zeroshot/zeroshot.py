import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_CODE_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.append(str(PROJECT_CODE_DIR))

from finetune.configs.config import Config as FinetuneConfig
from finetune.modules.dataset import ECGDataset
from finetune.modules.metrics import compute_auc
from pretrain.configs.config import Config as PretrainConfig
from pretrain.modules.model import ECGPretrainModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"
def load_prompt_bank(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)
    prompt_bank = {}
    for label, text in raw_prompts.items():
        if isinstance(text, list):
            phrases = [t.strip() for t in text if t.strip()]
        else:
            phrases = [t.strip() for t in text.split(",") if t.strip()]
        prompt_bank[label.strip()] = phrases if phrases else [label.strip()]
    return prompt_bank


def build_class_text_features(model, labels, prompt_bank, device):
    class_features = []
    for label in labels:
        prompts = prompt_bank.get(label, [label])
        tokenizer_output = model.text_encoder._tokenize(prompts, device=device)
        with torch.no_grad():
            text_embeds = model.text_encoder(
                tokenizer_output.input_ids,
                tokenizer_output.attention_mask,
            )
            latents = model.clip.to_text_latent(text_embeds)
            latents = F.normalize(latents, dim=-1)
            class_feature = latents.mean(dim=0)
            class_feature = F.normalize(class_feature, dim=0)
        class_features.append(class_feature)
    return torch.stack(class_features, dim=0)


def encode_ecg_latents(model, ecg):
    ecg= model.clip.visual_transformer(ecg)
    latents = model.clip.to_visual_latent(ecg)
    
    return F.normalize(latents, dim=-1)


def load_pretrained_clip(device):
    pretrain_config = PretrainConfig()
    model = ECGPretrainModel(pretrain_config)
    ckpt_dir = Path(pretrain_config.checkpoint_dir)
    checkpoint_path = ckpt_dir / "20251114_025956" / "weights.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到预训练权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def evaluate_dataset(dataset_name, model, prompt_bank, finetune_config):
    device = torch.device(finetune_config.device)
    dataset = ECGDataset(finetune_config.dataset_root, dataset_name, mode="test")
    num_workers = 6
    loader = DataLoader(
        dataset,
        batch_size=finetune_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    class_text = build_class_text_features(model, dataset.labels_name, prompt_bank, device)
    temperature = torch.exp(model.clip.temperature).detach()
    labels_buffer, probs_buffer = [], []
    with torch.no_grad():
        for ecg, labels in loader:
            ecg = ecg.to(device, non_blocking=True)
            image_latents = encode_ecg_latents(model, ecg)
            logits = image_latents @ class_text.t()
            logits = logits * temperature
            probs = torch.sigmoid(logits)
            labels_buffer.append(labels.cpu().numpy())
            probs_buffer.append(probs.cpu().numpy())
    labels_arr = np.vstack(labels_buffer)
    probs_arr = np.vstack(probs_buffer)
    auc = compute_auc(labels_arr, probs_arr)
    return auc


def main():
    finetune_config = FinetuneConfig()
    device = torch.device(finetune_config.device)
    model = load_pretrained_clip(device)
    prompt_json = Path(__file__).with_name("CKEPE_prompt.json")
    prompt_bank = load_prompt_bank(prompt_json)
    dataset_names = [
        "ptbxl-super",
        "ptbxl-sub",
        "ptbxl-form",
        "ptbxl-rhythm",
        "icbeb",
        "chapman",
    ]
    results = {}
    for dataset_name in dataset_names:
        print("=" * 60)
        print(f"Zero-shot {dataset_name}")
        auc = evaluate_dataset(dataset_name, model, prompt_bank, finetune_config)
        results[dataset_name] = auc
        print(f"{dataset_name} AUC: {auc:.4f}")
    results_path = Path(__file__).with_name("zeroshot_auc.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print(f"AUC 结果已保存到 {results_path}")


if __name__ == "__main__":
    main()
