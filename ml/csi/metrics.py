"""Métricas de classificação binária, 3 classes e de eventos."""
from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred)); tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred)); fn = int(np.sum(y_true & ~y_pred))
    tpr = tp / (tp + fn) if tp + fn else float('nan')
    tnr = tn / (tn + fp) if tn + fp else float('nan')
    prec = tp / (tp + fp) if tp + fp else float('nan')
    f1 = 2 * prec * tpr / (prec + tpr) if tp else 0.0
    return dict(n=len(y_true), TP=tp, TN=tn, FP=fp, FN=fn, recall=tpr, specificity=tnr,
                precision=prec, f1=f1, balanced_accuracy=np.nanmean([tpr, tnr]),
                accuracy=(tp + tn) / max(len(y_true), 1))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        m[int(a), int(b)] += 1
    return m


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def episode_detected(pred_positive: np.ndarray, min_fraction: float = 0.5) -> bool:
    """Um episódio (fase) é considerado reconhecido se a maioria das janelas seguras acertou."""
    return bool(len(pred_positive)) and float(np.mean(pred_positive)) >= min_fraction
