"""Evaluation of the pneumonia classifier (M3, Drop 3).

Metrics: accuracy, precision, recall (= sensitivity), specificity, F1, AUROC + confusion matrix.
Positive class = PNEUMONIA (index 1).

    python -m src.diagnosis.evaluate --ckpt models/best_cnn.pth --data_root /kaggle/input/chest-xray-pneumonia \
        --out_dir results
Writes: results/confusion_matrix.png, roc_curve.png, cnn_test_metrics.json, classification_report.txt
        (+ training_curves.png if models/cnn_history.json exists)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

CLASSES = ["NORMAL", "PNEUMONIA"]


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def classification_metrics(y_true, prob_pos, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(prob_pos) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),          # sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, prob_pos) if len(np.unique(y_true)) > 1 else float("nan"),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def best_threshold(y_true, prob_pos) -> float:
    """Youden's J on the VALIDATION set (never tune on test)."""
    fpr, tpr, thr = roc_curve(y_true, prob_pos)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))


# --------------------------------------------------------------------------- plots
def plot_confusion_matrix(cm, path, classes=CLASSES, title="Confusion matrix (test)"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / max(cm[i].sum(), 1) * 100
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, prob_pos, path):
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    auc = roc_auc_score(y_true, prob_pos)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(fpr, tpr, lw=2, label=f"EfficientNet-B0 (AUROC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: list[dict], path, metric_keys=("val_acc", "val_auroc", "val_f1"),
                         title="EfficientNet-B0 training"):
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
    a1.plot(ep, [h["val_loss"] for h in history], "o-", label="val loss")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("Loss"); a1.legend()
    for k in metric_keys:
        if k in history[0]:
            a2.plot(ep, [h[k] for h in history], "o-", label=k.replace("val_", "val "))
    if "train_acc" in history[0]:
        a2.plot(ep, [h["train_acc"] for h in history], "o--", label="train acc")
    a2.set_xlabel("epoch"); a2.set_title("Validation metrics"); a2.legend(); a2.set_ylim(0, 1.02)
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.set_xticks(ep)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI
def main():
    from src.data.dataset import make_cls_loaders
    from src.diagnosis.classifier import load_classifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/best_cnn.pth")
    ap.add_argument("--data_root")
    ap.add_argument("--csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tune_threshold", action="store_true",
                    help="pick threshold on VAL (Youden J) and report test at that threshold too")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, ck = load_classifier(a.ckpt, dev)
    loaders, _, _ = make_cls_loaders(a.data_root, a.csv, batch_size=a.bs, num_workers=a.workers)

    y, p = predict_loader(model, loaders[a.split], dev)
    res = {"default_threshold": classification_metrics(y, p[:, 1], 0.5)}
    if a.tune_threshold:
        yv, pv = predict_loader(model, loaders["val"], dev)
        t = best_threshold(yv, pv[:, 1])
        res["val_tuned_threshold"] = classification_metrics(y, p[:, 1], t)

    main_res = res.get("val_tuned_threshold", res["default_threshold"])
    plot_confusion_matrix(main_res["confusion_matrix"], out / "confusion_matrix.png",
                          title=f"Confusion matrix ({a.split}, thr={main_res['threshold']:.3f})")
    plot_roc(y, p[:, 1], out / "roc_curve.png")
    (out / "cnn_test_metrics.json").write_text(json.dumps(res, indent=2))
    (out / "classification_report.txt").write_text(
        classification_report(y, (p[:, 1] >= main_res["threshold"]).astype(int), target_names=CLASSES, digits=4))

    hist = Path(a.ckpt).with_name("cnn_history.json")
    if hist.exists():
        plot_training_curves(json.loads(hist.read_text()), out / "training_curves.png")

    print(json.dumps(res, indent=2))
    acc = main_res["accuracy"]
    print(f"\nTarget accuracy >= 0.85 : {'PASS' if acc >= 0.85 else 'NOT MET'} ({acc:.4f})")


if __name__ == "__main__":
    main()
