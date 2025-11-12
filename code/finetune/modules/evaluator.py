import torch
from tqdm import tqdm
import numpy as np
import time
from .metrics import (
    compute_accuracy,
    compute_auc,
    compute_sensitivity_specificity,
    compute_macro_f1,
    compute_matthews_corrcoef
)


def test(model, dataloader, device):
    model.to(device).eval()
    labels_list, probs_list, preds_list = [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Test', 
                       bar_format='{l_bar}{bar:30}{r_bar}',
                       colour='green')
        
        for batch in pbar:
            ecg, labels = batch
            ecg = ecg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            logits = model(ecg)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            labels_list.append(labels.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())

    labels_arr = np.vstack(labels_list)
    probs_arr = np.vstack(probs_list)
    preds_arr = np.vstack(preds_list)

    accuracy = float(np.mean(np.all(preds_arr == labels_arr, axis=1)))
    auc = compute_auc(labels_arr, probs_arr)
    sens, spec = compute_sensitivity_specificity(labels_arr, preds_arr)
    f1 = compute_macro_f1(labels_arr, preds_arr)
    mcc = compute_matthews_corrcoef(labels_arr, preds_arr)
    return accuracy, auc, sens, spec, f1, mcc
