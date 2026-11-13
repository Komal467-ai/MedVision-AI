"""Train one model configuration and save everything the paper needs.

    python -m src.train --data_root /kaggle/input/... --arch resnet50 --seed 0
    python -m src.train --data_root ... --arch efficientnet_b0 --split_mode image   # E2 leakage
    python -m src.train --data_root ... --arch densenet121 --mask_mode lungs_removed  # E3 shortcut
    python -m src.train --data_root ... --arch resnet50 --no_clahe                   # E6 ablation

Same recipe for every architecture (fair comparison): AdamW, one-cycle LR (10% warm-up),
class-weighted cross-entropy, mixed precision, 224x224, best epoch chosen on VALIDATION macro-F1.
Only the peak learning rate differs by family (zoo.SPECS[arch].lr), as is standard practice.

Writes runs/<run_name>/: best.pth, config.json, history.json, preds_val.csv, preds_test.csv, metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.cxrdata.dataset import CLASSES, leakage_count, make_loaders, prepare_table
from src.models.zoo import ARCHS, SPECS, Classifier, count_params, save_checkpoint
from src.research.metrics import fit_temperature, selective_accuracy, softmax, summary


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run_name(a) -> str:
    tags = [a.arch, a.split_mode, a.mask_mode, f"s{a.seed}"]
    for flag in ("no_clahe", "no_augment", "no_pretrained", "no_class_weights"):
        if getattr(a, flag):
            tags.append(flag)
    return "_".join(tags)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    logits, ys, idx = [], [], []
    for x, y, i in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out = model(x.to(dev, non_blocking=True))
        if not torch.isfinite(out).all():
            out = model(x.to(dev).float())
        logits.append(torch.nan_to_num(out.float(), nan=0.0, posinf=50.0, neginf=-50.0).cpu().numpy()); ys.append(y.numpy()); idx.append(i.numpy())
    return np.concatenate(logits), np.concatenate(ys), np.concatenate(idx)


def save_preds(path, df_part, logits, y, idx, T):
    rows = df_part.reset_index(drop=True).iloc[idx]
    p = softmax(logits, 1.0)
    pt = softmax(logits, T)
    out = pd.DataFrame({"path": rows["path"].values, "y": y})
    for k, c in enumerate(CLASSES):
        out[f"logit_{k}"] = logits[:, k]
        out[f"p_{k}"] = p[:, k]
        out[f"pT_{k}"] = pt[:, k]
    out.to_csv(path, index=False)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    ap.add_argument("--split_mode", default="group", choices=["group", "image"])
    ap.add_argument("--mask_mode", default="none", choices=["none", "lungs_only", "lungs_removed", "lung_crop"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=42, help="kept fixed so every run sees the same test set")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: per-family value in zoo.SPECS")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--no_clahe", action="store_true")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--max_batches", type=int, default=0, help=">0 = smoke test")
    ap.add_argument("--no_dedup", action="store_true", help="skip duplicate hashing (smoke tests only)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.runs_dir) / run_name(a)
    out.mkdir(parents=True, exist_ok=True)

    df = prepare_table(a.data_root, a.split_mode, seed=a.split_seed, cache_dir=str(Path(a.runs_dir) / "cache"),
                       dedup=not a.no_dedup)
    loaders, cw = make_loaders(df, a.img_size, a.bs, a.workers, a.mask_mode, clahe=not a.no_clahe,
                               augment=not a.no_augment)
    model = Classifier(a.arch, len(CLASSES), pretrained=not a.no_pretrained).to(dev)
    lr = a.lr or SPECS[a.arch].lr
    config = {**vars(a), "lr_used": lr, "device": dev, "params": count_params(model),
              "split_counts": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
              "duplicate_groups_crossing_splits": leakage_count(df)}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"[{out.name}] {a.arch}: {config['params']/1e6:.2f}M params | lr {lr} | device {dev} | "
          f"leaky groups {config['duplicate_groups_crossing_splits']}")

    crit = nn.CrossEntropyLoss(weight=None if a.no_class_weights else cw.to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.wd)
    steps_per_ep = min(len(loaders["train"]), a.max_batches) if a.max_batches else len(loaders["train"])
    total = max(a.epochs * steps_per_ep, 4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=max(0.1, 2 / total))
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tl = n = correct = 0
        for b, (x, y, _) in enumerate(loaders["train"]):
            if a.max_batches and b >= a.max_batches:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); n += len(y); correct += (logits.argmax(1) == y).sum().item()
        vl, vy, _ = predict(model, loaders["val"], dev)
        m = summary(vy, softmax(vl))
        rec = {"epoch": ep, "train_loss": tl / max(n, 1), "train_acc": correct / max(n, 1),
               "val_loss": float(nn.functional.cross_entropy(torch.tensor(vl), torch.tensor(vy)).item()),
               "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_macro_auroc": m["macro_auroc"],
               "time_s": round(time.time() - t0, 1)}
        hist.append(rec)
        flag = ""
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            save_checkpoint(model, out / "best.pth", a.img_size, recipe=config, val_metrics=m)
            flag = "  <- best"
        print(f"ep {ep}/{a.epochs} | loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
              f"val acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} AUROC {m['macro_auroc']:.4f} | "
              f"{rec['time_s']}s{flag}")
    (out / "history.json").write_text(json.dumps(hist, indent=2))

    # ---- final evaluation with the best checkpoint
    ck = torch.load(out / "best.pth", map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    vl, vy, vi = predict(model, loaders["val"], dev)
    tl_, ty, ti = predict(model, loaders["test"], dev)
    T = fit_temperature(vl, vy)  # calibration fitted on validation only
    save_preds(out / "preds_val.csv", loaders["val"].dataset.df, vl, vy, vi, T)
    save_preds(out / "preds_test.csv", loaders["test"].dataset.df, tl_, ty, ti, T)
    res = {"run": out.name, "arch": a.arch, "family": SPECS[a.arch].family, "split_mode": a.split_mode,
           "mask_mode": a.mask_mode, "seed": a.seed, "temperature": round(T, 4),
           "val": summary(vy, softmax(vl)), "test": summary(ty, softmax(tl_)),
           "test_calibrated_ece": summary(ty, softmax(tl_, T))["ece"],
           "test_selective_accuracy": selective_accuracy(ty, softmax(tl_, T))}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    t = res["test"]
    print(f"TEST {out.name}: acc {t['accuracy']} | macro-F1 {t['macro_f1']} | macro-AUROC {t['macro_auroc']} | "
          f"ECE {t['ece']} -> {res['test_calibrated_ece']} after T={T:.2f}")
    return res


if __name__ == "__main__":
    main()
