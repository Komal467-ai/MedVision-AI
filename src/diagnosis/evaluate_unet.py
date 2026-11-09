"""Evaluate U-Net on the held-out TEST patients + overlay figure (M3, Drop 4).

    python -m src.diagnosis.evaluate_unet --ckpt models/best_unet.pth --img_dir ... --mask_dir ...
    python -m src.diagnosis.evaluate_unet --ckpt models/best_unet.pth --seg_root /kaggle/input/lgg-mri-segmentation/kaggle_3m
Writes: results/unet_test_metrics.json, results/segmentation_overlay.png
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

from src.diagnosis.losses import DiceBCELoss, dice_iou_per_image
from src.diagnosis.train_unet import run_eval
from src.diagnosis.unet import load_unet
from src.preprocessing.augment import denormalize


def colour_overlay(rgb01, gt, pred):
    """green = ground truth only, red = prediction only, yellow = overlap (true positive)."""
    out = rgb01.copy()
    gt, pred = gt.astype(bool), pred.astype(bool)
    for m, c in ((gt & ~pred, (0, 1, 0)), (pred & ~gt, (1, 0, 0)), (gt & pred, (1, 1, 0))):
        out[m] = 0.5 * out[m] + 0.5 * np.array(c)
    return out


@torch.no_grad()
def overlay_figure(model, dataset, out_path, n=4, dev="cpu", seed=0):
    """Pick n test slices that contain tumour, show image | GT | prediction | overlay."""
    rng = np.random.default_rng(seed)
    pos = [i for i in range(len(dataset)) if dataset[i][1].sum() > 0] or list(range(len(dataset)))
    idx = rng.choice(pos, size=min(n, len(pos)), replace=False)
    fig, axes = plt.subplots(len(idx), 4, figsize=(12, 3.1 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        x, y = dataset[int(i)]
        logit = model(x[None].to(dev))
        pred = (torch.sigmoid(logit)[0, 0] > 0.5).cpu().numpy()
        d, iou = dice_iou_per_image(logit.cpu(), y[None])
        img, gt = denormalize(x), y[0].numpy() > 0.5
        for ax, im, t in zip(axes[r], [img, gt, pred, colour_overlay(img, gt, pred)],
                             ["MRI slice", "Ground truth", "U-Net prediction",
                              f"Overlay  Dice {d.item():.3f}  IoU {iou.item():.3f}"]):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray"); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.suptitle("U-Net segmentation (test patients) - green: GT only, red: false positive, yellow: overlap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    from src.data.dataset import make_seg_loaders

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/best_unet.pth")
    ap.add_argument("--seg_root")
    ap.add_argument("--img_dir")
    ap.add_argument("--mask_dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42, help="must match the training seed (same split)")
    ap.add_argument("--n_overlay", type=int, default=4)
    ap.add_argument("--out_dir", default="results")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, ck = load_unet(a.ckpt, dev)
    loaders, _ = make_seg_loaders(a.seg_root, a.img_dir, a.mask_dir, a.bs, a.workers, a.seed)
    loss, m = run_eval(model, loaders[a.split], DiceBCELoss(), dev)
    m["loss"] = round(loss, 4)
    (out / "unet_test_metrics.json").write_text(json.dumps(m, indent=2))
    overlay_figure(model, loaders[a.split].dataset, out / "segmentation_overlay.png", a.n_overlay, dev)
    print(json.dumps(m, indent=2))
    print(f"\nTarget Dice >= 0.80 : {'PASS' if m['dice_mean'] >= 0.80 else 'NOT MET'} ({m['dice_mean']:.4f})")


if __name__ == "__main__":
    main()
