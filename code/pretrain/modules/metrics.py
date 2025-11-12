import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score


def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return accuracy_score(labels, preds)
    
def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    return roc_auc_score(labels, probs, average='macro')

def compute_sensitivity_specificity(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    sens, spec = [], []
    for i in range(labels.shape[1]):
        col, pcol = labels[:, i], preds[:, i]
        tp = np.sum((col == 1) & (pcol == 1))
        tn = np.sum((col == 0) & (pcol == 0))
        fp = np.sum((col == 0) & (pcol == 1))
        fn = np.sum((col == 1) & (pcol == 0))
        sens.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        spec.append(tn / (tn + fp) if tn + fp > 0 else 0.0)
    return float(np.mean(sens)), float(np.mean(spec))

def compute_macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    return f1_score(labels, preds, average='macro', zero_division=0)

def compute_matthews_corrcoef(labels: np.ndarray, preds: np.ndarray) -> float:
    scores = []
    for i in range(labels.shape[1]):
        scores.append(matthews_corrcoef(labels[:, i], preds[:, i]))
    return float(np.mean(scores)) if scores else 0.0

