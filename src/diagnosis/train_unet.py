"""Train the U-Net tumour segmenter (M3, Drop 4).

Layout A (M2's BraTS PNG slices):
    python -m src.diagnosis.train_unet --img_dir data/seg/images --mask_dir data/seg/masks --epochs 25
Layout B (Kaggle 'lgg-mri-segmentation', backup dataset):
    python -m src.diagnosis.train_unet --seg_root /kaggle/input/lgg-mri-segmentation/kaggle_3m --epochs 25

Recipe: AdamW lr 1e-3 (wd 1e-4), cosine schedule, Dice+BCE loss, AMP, 256x256,
patient-level 70/15/15 split, best checkpoint by VALIDATION Dice.
Outputs: models/best_unet.pth, models/unet_history.json, results/unet_training_curves.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.dataset import make_seg_loaders
from src.diagnosis.evaluate import plot_training_curves
from src.diagnosis.losses import DiceBCELoss, SegMeter
from src.diagnosis.train_classifier import set_seed
from src.diagnosis.unet import UNet, save_unet


@torch.no_grad()
def run_eval(model, loader, crit, dev, max_batches=0):
    model.eval()
    meter, tot, n = SegMeter(), 0.0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        tot += crit(out, y).item() * len(x); n += len(x)
        meter.update(out, y)
    return tot / max(n, 1), meter.compute()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_root")
    ap.add_argument("--img_dir")
    ap.add_argument("--mask_dir")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32, help="U-Net base width (32 -> ~7.8M params)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_positive_train", action="store_true",
                    help="train only on slices containing tumour (val/test keep all slices)")
    ap.add_argument("--out", default="models/best_unet.pth")
    ap.add_argument("--results", default="results")
    ap.add_argument("--max_batches", type=int, default=0, help=">0: smoke test")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.results).mkdir(parents=True, exist_ok=True)

    loaders, df = make_seg_loaders(a.seg_root, a.img_dir, a.mask_dir, a.bs, a.workers, a.seed,
                                   a.only_positive_train)
    print(df.groupby("split").agg(images=("image", "size"), patients=("patient_id", "nunique")))

    model = UNet(base=a.base).to(dev)
    print(f"U-Net: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params | device {dev}")
    crit = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tl, n = 0.0, 0
        for b, (x, y) in enumerate(loaders["train"]):
            if a.max_batches and b >= a.max_batches:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(x); n += len(x)
        sched.step()

        vl, m = run_eval(model, loaders["val"], crit, dev, a.max_batches)
        rec = {"epoch": ep, "train_loss": tl / max(n, 1), "val_loss": vl, "val_dice": m["dice_mean"],
               "val_iou": m["iou_mean"], "val_dice_tumour": m["dice_tumour_slices"],
               "lr": opt.param_groups[0]["lr"], "time_s": round(time.time() - t0, 1)}
        history.append(rec)
        flag = ""
        if m["dice_mean"] > best:
            best = m["dice_mean"]
            save_unet(model, a.out, epoch=ep, val_metrics=m)
            flag = "  <- saved best"
        print(f"ep {ep}/{a.epochs} | train {rec['train_loss']:.4f} | val loss {vl:.4f} Dice {m['dice_mean']:.4f} "
              f"IoU {m['iou_mean']:.4f} Dice(tumour slices) {m['dice_tumour_slices']:.4f} | {rec['time_s']}s{flag}")

    hist = Path(a.out).with_name("unet_history.json")
    hist.write_text(json.dumps(history, indent=2))
    plot_training_curves(history, Path(a.results) / "unet_training_curves.png",
                         metric_keys=("val_dice", "val_iou", "val_dice_tumour"), title="U-Net training")
    print(f"best val Dice {best:.4f} -> {a.out}")
    return history


if __name__ == "__main__":
    main()
