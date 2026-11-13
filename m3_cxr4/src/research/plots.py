"""Figures for the paper, drawn from runs/ and results/ (run after aggregate.py).

    python -m src.research.plots --runs_dir runs --results results

fig_model_comparison.png   macro-F1 per model (mean +/- SD over seeds) + ensemble
fig_confusion_best.png     4x4 confusion matrix of the best configuration
fig_roc_best.png           one-vs-rest ROC per disease for the best configuration
fig_reliability.png        calibration before / after temperature scaling (best single model)
fig_leakage.png            group vs image split per model
fig_shortcut.png           full image vs lungs_only vs lungs_removed vs lung_crop per model
fig_per_class_sens.png     sensitivity per disease per model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.cxrdata.dataset import CLASSES, SHORT

LABELS = {"none": "full image", "lungs_only": "lungs only", "lungs_removed": "lungs removed",
          "lung_crop": "lung crop", "group": "duplicate-aware split", "image": "image-level split"}
ORDER = ["resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "vit_b_16", "swin_t", "ensemble"]


def _order(names):
    return sorted(names, key=lambda n: ORDER.index(n) if n in ORDER else 99)


def bar_by(df, x_col, group_col, groups, metric, title, path, ylabel="Macro-F1 (test)"):
    archs = _order(df[x_col].unique())
    w = 0.8 / max(len(groups), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(archs)), 3.8))
    for i, gname in enumerate(groups):
        sub = df[df[group_col] == gname].groupby(x_col)[metric].agg(["mean", "std"])
        means = [sub["mean"].get(a, np.nan) for a in archs]
        sds = [sub["std"].get(a, 0) if not np.isnan(sub["std"].get(a, np.nan)) else 0 for a in archs]
        ax.bar(np.arange(len(archs)) + (i - (len(groups) - 1) / 2) * w, means, w, yerr=sds, capsize=3, label=LABELS.get(gname, gname))
    ax.set_xticks(range(len(archs)), archs, rotation=20)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    lo = np.nanmin(df[metric]) if len(df) else 0
    ax.set_ylim(max(0, lo - 0.1), 1.0)
    if len(groups) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--main_ablation", default="none")
    a = ap.parse_args(argv)
    rd, res = Path(a.runs_dir), Path(a.results)
    df = pd.read_csv(res / "all_runs.csv")
    main_df = df[(df.split_mode == "group") & (df.mask_mode == "none") & (df.ablation == a.main_ablation)]

    if len(main_df):
        bar_by(main_df.assign(g="all"), "arch", "g", ["all"], "macro_f1",
               "Model comparison (group split, mean ± SD over seeds)", res / "fig_model_comparison.png")
    leak = df[(df.mask_mode == "none") & (df.ablation == a.main_ablation) & (df.arch != "ensemble")]
    if leak.split_mode.nunique() > 1:
        bar_by(leak, "arch", "split_mode", ["group", "image"], "accuracy",
               "Leakage: duplicate-aware vs image-level split", res / "fig_leakage.png", "Accuracy (test)")
    sc = df[(df.split_mode == "group") & (df.ablation == a.main_ablation) & (df.arch != "ensemble")]
    if sc.mask_mode.nunique() > 1:
        modes = [m for m in ["none", "lungs_only", "lung_crop", "lungs_removed"] if m in sc.mask_mode.unique()]
        bar_by(sc, "arch", "mask_mode", modes, "macro_f1", "Shortcut audit: what the model is allowed to see",
               res / "fig_shortcut.png")

    # per-class sensitivity
    if len(main_df):
        s = main_df.groupby("arch")[[f"sens_{c}" for c in CLASSES]].mean().loc[_order(main_df.arch.unique())]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for k, c in enumerate(CLASSES):
            ax.plot(s.index, s[f"sens_{c}"], "o-", label=SHORT[k])
        ax.set_ylabel("Sensitivity (test)"); ax.set_title("Per-disease sensitivity"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8); plt.xticks(rotation=20); fig.tight_layout()
        fig.savefig(res / "fig_per_class_sens.png", dpi=150); plt.close(fig)

    # best configuration: confusion matrix + ROC + reliability
    if len(main_df):
        best = main_df.sort_values("macro_f1", ascending=False).iloc[0]["run"]
        d = pd.read_csv(rd / best / "preds_test.csv")
        y = d["y"].values
        p = d[[f"pT_{k}" for k in range(len(CLASSES))]].values
        m = json.loads((rd / best / "metrics.json").read_text())
        cm = np.array(m["test"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4.4))
        ax.imshow(cm, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / max(cm[i].sum(), 1) * 100:.1f}%", ha="center", va="center",
                        fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(4), SHORT); ax.set_yticks(range(4), SHORT)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix - {best}", fontsize=9)
        fig.tight_layout(); fig.savefig(res / "fig_confusion_best.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4.4))
        for k in range(4):
            fpr, tpr, _ = roc_curve((y == k).astype(int), p[:, k])
            ax.plot(fpr, tpr, label=f"{SHORT[k]} (AUROC {m['test']['per_class'][CLASSES[k]]['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"One-vs-rest ROC - {best}", fontsize=9); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(res / "fig_roc_best.png", dpi=150); plt.close(fig)

        single = main_df[main_df.arch != "ensemble"].sort_values("macro_f1", ascending=False)
        if len(single):
            r = single.iloc[0]["run"]
            d = pd.read_csv(rd / r / "preds_test.csv")
            y = d["y"].values
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            for col, lab in (("p_", "uncalibrated"), ("pT_", "temperature-scaled")):
                pr = d[[f"{col}{k}" for k in range(4)]].values
                conf, ok = pr.max(1), pr.argmax(1) == y
                bins = np.linspace(0, 1, 11)
                idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
                xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
                ys = [ok[idx == b].mean() for b in range(10) if (idx == b).any()]
                ax.plot(xs, ys, "o-", label=lab)
            ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
            ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title(f"Reliability - {r}", fontsize=9)
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(res / "fig_reliability.png", dpi=150); plt.close(fig)
    print("figures written to", res)


if __name__ == "__main__":
    main()
