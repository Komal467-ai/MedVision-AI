"""Train EfficientNet-B0 pneumonia classifier (M3, Drop 3).

Kaggle (GPU T4/P100, ~25-35 min for 5 epochs):
    python -m src.diagnosis.train_classifier --data_root /kaggle/input/chest-xray-pneumonia/chest_xray \
        --epochs 5 --out models/best_cnn.pth
Or with M2's split CSV:
    python -m src.diagnosis.train_classifier --csv data/splits/cxr_split.csv --data_root data/raw

Recipe: AdamW (lr 3e-4, wd 1e-4), cosine LR schedule, class-weighted CrossEntropyLoss,
mixed precision on GPU, best checkpoint chosen on VALIDATION AUROC (threshold-free and stable).
Outputs: models/best_cnn.pth, models/cnn_history.json, results/training_curves.png
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import make_cls_loaders
from src.diagnosis.classifier import build_classifier, count_parameters, save_checkpoint
from src.diagnosis.evaluate import classification_metrics, plot_training_curves, predict_loader


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root")
    ap.add_argument("--csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/best_cnn.pth")
    ap.add_argument("--results", default="results")
    ap.add_argument("--no_pretrained", action="store_true", help="random init (offline smoke test only)")
    ap.add_argument("--max_batches", type=int, default=0, help=">0: limit batches/epoch (smoke test)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.results).mkdir(parents=True, exist_ok=True)

    loaders, cw, df = make_cls_loaders(a.data_root, a.csv, a.bs, a.workers, a.seed)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0), "\nclass weights:", cw.tolist())

    model = build_classifier(2, pretrained=not a.no_pretrained).to(dev)
    tot, tr = count_parameters(model)
    print(f"EfficientNet-B0: {tot/1e6:.2f}M params ({tr/1e6:.2f}M trainable)")

    crit = nn.CrossEntropyLoss(weight=cw.to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (min(len(loaders["train"]), a.max_batches) if a.max_batches else len(loaders["train"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = correct = seen = 0
        for b, (x, y) in enumerate(loaders["train"]):
            if a.max_batches and b >= a.max_batches:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        # ---- validation
        yv, pv = predict_loader(model, loaders["val"], dev)
        with torch.no_grad():
            val_loss = nn.functional.cross_entropy(torch.log(torch.tensor(pv).clamp_min(1e-7)),
                                                   torch.tensor(yv), weight=cw).item()
        m = classification_metrics(yv, pv[:, 1])
        rec = {"epoch": ep, "train_loss": run_loss / seen, "train_acc": correct / seen, "val_loss": val_loss,
               "val_acc": m["accuracy"], "val_f1": m["f1"], "val_auroc": m["auroc"],
               "val_recall": m["recall"], "val_specificity": m["specificity"],
               "lr": opt.param_groups[0]["lr"], "time_s": round(time.time() - t0, 1)}
        history.append(rec)
        score = m["auroc"] if not np.isnan(m["auroc"]) else m["accuracy"]
        flag = ""
        if score > best:
            best = score
            save_checkpoint(model, a.out, epoch=ep, val_metrics=m)
            flag = "  <- saved best"
        print(f"ep {ep}/{a.epochs} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
              f"val loss {val_loss:.4f} acc {m['accuracy']:.4f} F1 {m['f1']:.4f} AUROC {m['auroc']:.4f} "
              f"| {rec['time_s']}s{flag}")

    hist_path = Path(a.out).with_name("cnn_history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    plot_training_curves(history, Path(a.results) / "training_curves.png")
    print(f"best val AUROC {best:.4f} -> {a.out}\nhistory -> {hist_path}")
    return history


if __name__ == "__main__":
    main()
