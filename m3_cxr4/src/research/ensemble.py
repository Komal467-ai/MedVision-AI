"""Soft-voting ensembles of trained runs (the 'mix of models').

    python -m src.research.ensemble --runs runs/resnet50_group_none_s0 runs/densenet121_group_none_s0 ... --name all6_s0
    python -m src.research.ensemble --pattern "*_group_none_s0" --greedy --name greedy_s0

Average = mean of the temperature-calibrated class probabilities of each member.
--greedy: forward selection of members on VALIDATION macro-F1 (Caruana et al., 2004), so the
test set is never used to choose the ensemble.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cxrdata.dataset import CLASSES
from src.research.metrics import selective_accuracy, summary

PCOLS = [f"pT_{k}" for k in range(len(CLASSES))]


def load_preds(run: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(run / f"preds_{split}.csv").sort_values("path").reset_index(drop=True)


def average(runs: list[Path], split: str):
    frames = [load_preds(r, split) for r in runs]
    ref = frames[0]
    for f in frames[1:]:
        if not (f["path"].values == ref["path"].values).all():
            raise ValueError("runs were evaluated on different images - use the same split_mode/split_seed")
    probs = np.mean([f[PCOLS].values for f in frames], axis=0)
    return ref["y"].values, probs, ref["path"].values


def greedy_select(cands: list[Path], max_members: int = 6) -> list[Path]:
    chosen, best = [], -1.0
    improved = True
    while improved and len(chosen) < max_members:
        improved = False
        for c in cands:
            trial = chosen + [c]
            y, p, _ = average(trial, "val")
            f1 = summary(y, p)["macro_f1"]
            if f1 > best + 1e-4:
                best, pick, improved = f1, c, True
        if improved:
            chosen.append(pick)
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--pattern", help="glob under runs_dir, e.g. '*_group_none_s0'")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)

    runs = [Path(r) for r in a.runs]
    if a.pattern:
        runs += sorted(p for p in Path(a.runs_dir).glob(a.pattern) if (p / "preds_test.csv").exists())
    if len(runs) < 2:
        raise SystemExit("need at least two runs")
    members = greedy_select(runs) if a.greedy else runs

    yv, pv, _ = average(members, "val")
    yt, pt, paths = average(members, "test")
    out = Path(a.runs_dir) / f"ensemble_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"path": paths, "y": yt})
    for k in range(len(CLASSES)):
        df[f"p_{k}"] = df[f"pT_{k}"] = pt[:, k]
    df.to_csv(out / "preds_test.csv", index=False)
    res = {"run": out.name, "arch": "ensemble", "family": "Ensemble", "members": [m.name for m in members],
           "val": summary(yv, pv), "test": summary(yt, pt), "test_selective_accuracy": selective_accuracy(yt, pt)}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    t = res["test"]
    print(f"ENSEMBLE {a.name} ({len(members)}: {[m.name.split('_')[0] for m in members]}) -> "
          f"acc {t['accuracy']} macro-F1 {t['macro_f1']} macro-AUROC {t['macro_auroc']}")
    return res


if __name__ == "__main__":
    main()
