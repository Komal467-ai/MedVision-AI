"""Turn all runs into the paper's tables (mean +/- SD over seeds, 95% CIs, significance tests).

    python -m src.research.aggregate --runs_dir runs --out results

Writes:
  results/all_runs.csv           one row per run (test metrics)
  results/table_main.csv / .md   per configuration: mean +/- SD over seeds
  results/table_ci.csv           bootstrap 95% CI (seed-0 run or ensemble) for macro-F1 and macro-AUROC
  results/pairwise_mcnemar.csv   every pair of seed-0 baseline models + ensembles, Holm-corrected
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from src.cxrdata.dataset import CLASSES
from src.research.metrics import bootstrap_ci

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "ece"]


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for mf in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(mf.read_text())
        cfg_p = mf.parent / "config.json"
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        ablation = "+".join(f for f in ("no_clahe", "no_augment", "no_pretrained", "no_class_weights") if cfg.get(f))
        row = {"run": m["run"], "arch": m["arch"], "family": m.get("family", ""),
               "split_mode": m.get("split_mode", "group"), "mask_mode": m.get("mask_mode", "none"),
               "ablation": ablation or "none", "seed": m.get("seed", -1),
               "ece_calibrated": m.get("test_calibrated_ece", m["test"]["ece"])}
        row.update({k: m["test"][k] for k in METRICS})
        for c in CLASSES:
            row[f"sens_{c}"] = m["test"]["per_class"][c]["sensitivity"]
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar(y, pa, pb) -> float:
    """Exact McNemar test (binomial on discordant pairs) when small, chi-square with continuity otherwise."""
    a_right, b_right = pa == y, pb == y
    n01, n10 = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    if n01 + n10 == 0:
        return 1.0
    if n01 + n10 < 25:
        return float(binomtest(n01, n01 + n10, 0.5).pvalue)
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(chi2.sf(stat, 1))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def preds(run_dir: Path):
    d = pd.read_csv(run_dir / "preds_test.csv").sort_values("path")
    p = d[[f"pT_{k}" for k in range(len(CLASSES))]].values
    return d["y"].values, p


def fmt(mean, sd):
    return f"{mean:.4f} ± {sd:.4f}" if not np.isnan(sd) else f"{mean:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--main_ablation", default="none", help="ablation tag of the main protocol runs")
    a = ap.parse_args(argv)
    rd, out = Path(a.runs_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs(rd)
    if df.empty:
        raise SystemExit("no runs found")
    df.to_csv(out / "all_runs.csv", index=False)

    keys = ["arch", "family", "split_mode", "mask_mode", "ablation"]
    g = df.groupby(keys)
    table = g[METRICS + ["ece_calibrated"]].agg(["mean", "std"])
    table["n_seeds"] = g.size()
    table = table.reset_index()
    table.columns = ["_".join(c).strip("_") for c in table.columns]
    table = table.sort_values("macro_f1_mean", ascending=False)
    table.to_csv(out / "table_main.csv", index=False)

    md = ["| Model | Family | Split | Input | Ablation | Seeds | Accuracy | Macro-F1 | Macro-AUROC | ECE → calibrated |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        md.append(f"| {r['arch']} | {r['family']} | {r['split_mode']} | {r['mask_mode']} | {r['ablation']} | "
                  f"{int(r['n_seeds'])} | {fmt(r['accuracy_mean'], r['accuracy_std'])} | "
                  f"{fmt(r['macro_f1_mean'], r['macro_f1_std'])} | {fmt(r['macro_auroc_mean'], r['macro_auroc_std'])} | "
                  f"{r['ece_mean']:.3f} → {r['ece_calibrated_mean']:.3f} |")
    (out / "table_main.md").write_text("\n".join(md) + "\n")

    # ---- CIs + pairwise tests on the main protocol (group split, full image, no ablation, seed 0) + ensembles
    base = df[(df.split_mode == "group") & (df.mask_mode == "none") & (df.ablation == a.main_ablation) &
              ((df.seed == 0) | (df.arch == "ensemble"))]
    ci_rows, cache = [], {}
    for _, r in base.iterrows():
        y, p = preds(rd / r["run"])
        cache[r["run"]] = (y, p)
        ci_rows.append({"run": r["run"], "macro_f1": r["macro_f1"], "macro_f1_ci": bootstrap_ci(y, p, "macro_f1"),
                        "macro_auroc": r["macro_auroc"], "macro_auroc_ci": bootstrap_ci(y, p, "macro_auroc")})
    pd.DataFrame(ci_rows).to_csv(out / "table_ci.csv", index=False)

    pairs = []
    for ra, rb in itertools.combinations(cache, 2):
        (ya, pa), (yb, pb) = cache[ra], cache[rb]
        if len(ya) != len(yb) or not (ya == yb).all():
            continue
        pairs.append({"a": ra, "b": rb, "acc_a": float((pa.argmax(1) == ya).mean()),
                      "acc_b": float((pb.argmax(1) == yb).mean()), "p_raw": mcnemar(ya, pa.argmax(1), pb.argmax(1))})
    if pairs:
        pw = pd.DataFrame(pairs)
        pw["p_holm"] = holm(pw["p_raw"].tolist())
        pw["significant_0.05"] = pw["p_holm"] < 0.05
        pw.to_csv(out / "pairwise_mcnemar.csv", index=False)
    print((out / "table_main.md").read_text())
    print(f"{len(df)} runs aggregated -> {out}")


if __name__ == "__main__":
    main()
