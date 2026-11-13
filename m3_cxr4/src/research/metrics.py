"""Multi-class metrics, calibration and confidence intervals.

All functions take y (int labels, shape [N]) and probs (shape [N, C]).
Macro averages weight each disease equally - important because Viral Pneumonia
(1,345 images) is 7x rarer than Normal (10,192).
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.cxrdata.dataset import CLASSES


def ece(y, probs, n_bins: int = 15) -> float:
    """Expected Calibration Error: gap between confidence and accuracy, averaged over bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def per_class(y, probs) -> dict:
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    out = {}
    for k, c in enumerate(CLASSES):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        yk = (y == k).astype(int)
        out[c] = {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                  "precision": tp / max(tp + fp, 1),
                  "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                  "auroc": roc_auc_score(yk, probs[:, k]) if 0 < yk.sum() < len(yk) else float("nan"),
                  "support": int(yk.sum())}
    return {c: {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v for k, v in d.items()}
            for c, d in out.items()}


def macro_auroc(y, probs) -> float:
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=range(probs.shape[1])))
    except ValueError:
        vals = [roc_auc_score((y == k).astype(int), probs[:, k]) for k in present]
        return float(np.mean(vals))


def summary(y, probs) -> dict:
    y, probs = np.asarray(y), np.asarray(probs, dtype=float)
    probs = np.nan_to_num(probs, nan=1.0 / probs.shape[1])
    pred = probs.argmax(1)
    m = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
        "macro_precision": precision_score(y, pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
        "macro_auroc": macro_auroc(y, probs),
        "ece": ece(y, probs),
        "n": int(len(y)),
    }
    m = {k: round(float(v), 4) if k != "n" else v for k, v in m.items()}
    m["per_class"] = per_class(y, probs)
    m["confusion_matrix"] = confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()
    return m


# ---------------------------------------------------------------- calibration
def fit_temperature(logits, y, iters: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T fitted on VALIDATION logits."""
    lg = torch.tensor(np.nan_to_num(np.asarray(logits, dtype=float), nan=0.0, posinf=50.0, neginf=-50.0), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        return 1.0
    T = float(log_t.exp().item())
    return T if np.isfinite(T) and 0.05 <= T <= 20 else 1.0


def softmax(logits, T: float = 1.0):
    z = np.asarray(logits) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------- uncertainty in the estimate
def bootstrap_ci(y, probs, metric: str = "macro_f1", n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """95% percentile bootstrap CI of a summary() metric over test images."""
    y, probs = np.asarray(y), np.asarray(probs)
    rng = np.random.default_rng(seed)
    vals = []
    warnings.filterwarnings("ignore", message="Only one class is present")
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        pred = probs[idx].argmax(1)
        if metric == "macro_f1":
            v = f1_score(y[idx], pred, average="macro", labels=range(probs.shape[1]), zero_division=0)
        elif metric == "accuracy":
            v = accuracy_score(y[idx], pred)
        elif metric == "balanced_accuracy":
            v = balanced_accuracy_score(y[idx], pred)
        elif metric == "macro_auroc":
            v = macro_auroc(y[idx], probs[idx])
        else:
            raise ValueError(metric)
        vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


def selective_accuracy(y, probs, coverages=(1.0, 0.95, 0.9, 0.8)) -> dict:
    """Accuracy when the model may abstain on its least-confident cases (refer to a doctor)."""
    y, probs = np.asarray(y), np.asarray(probs)
    conf = probs.max(1)
    order = np.argsort(-conf)
    out = {}
    for c in coverages:
        k = max(int(round(c * len(y))), 1)
        keep = order[:k]
        out[f"{int(c * 100)}%"] = round(float((probs[keep].argmax(1) == y[keep]).mean()), 4)
    return out
