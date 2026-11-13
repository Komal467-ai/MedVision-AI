"""Run the whole study (or one stage of it) with one command - used by the Kaggle notebook.

    python -m src.run_study --stage smoke              # every step, a few batches (~15 min on T4)
    python -m src.run_study --stage main --seeds 0 1 2 # E1 + E2 + E4
    python -m src.run_study --stage audits --seeds 0 1 # E3 + E5 + E7
    python -m src.run_study --stage analysis           # tables + figures (merges runs from attached inputs)

Finished runs (runs/<name>/metrics.json exists) are skipped, so a stage can be resumed.
--hours stops launching new runs once the time budget is used, so a 12-hour Kaggle session
always ends with its outputs saved.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ARCHS = ["resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "vit_b_16", "swin_t"]
AUDIT_ARCHS = ["resnet50", "efficientnet_b0", "swin_t"]
MASKS = ["lungs_only", "lungs_removed", "lung_crop"]
ABLATIONS = ["--no_clahe", "--no_augment", "--no_class_weights", "--no_pretrained"]
T0 = time.time()


def find_root(arg: str | None) -> str:
    if arg:
        return arg
    import kagglehub
    return kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")


def name_of(arch, seed, split="group", mask="none", extra=()):
    tags = [arch, split, mask, f"s{seed}"] + [e.lstrip("-") for e in extra]
    return "_".join(tags)


def py(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)


class Runner:
    def __init__(self, a):
        self.a, self.done, self.failed, self.skipped = a, [], [], []

    def out_of_time(self):
        return self.a.hours and (time.time() - T0) / 3600 > self.a.hours

    def train(self, arch, seed, split="group", mask="none", extra=()):
        extra = list(extra) + (["--no_pretrained"] if self.a.offline and "--no_pretrained" not in extra else [])
        name = name_of(arch, seed, split, mask, extra)
        if Path(self.a.runs_dir, name, "metrics.json").exists():
            self.skipped.append(name); return
        if self.out_of_time():
            print(f"[time budget reached] not starting {name}"); return
        cmd = ["src.train", "--data_root", self.a.root, "--arch", arch, "--seed", str(seed), "--split_mode", split,
               "--mask_mode", mask, "--epochs", str(self.a.epochs), "--workers", str(self.a.workers),
               "--runs_dir", self.a.runs_dir, "--img_size", str(self.a.img_size), "--bs", str(self.a.bs), *extra]
        if self.a.smoke:
            cmd += ["--max_batches", "5"]
        t = time.time()
        r = py(*cmd)
        mins = (time.time() - t) / 60
        test = [l for l in r.stdout.splitlines() if l.startswith("TEST")]
        if r.returncode or not test:
            self.failed.append(name)
            print(f"FAILED {name} ({mins:.1f} min)\n{r.stderr[-2500:]}")
        else:
            self.done.append(name)
            print(f"{test[-1]}  ({mins:.1f} min, total {(time.time() - T0) / 3600:.2f} h)")
        sys.stdout.flush()


def stage_main(R, seeds):
    for s in seeds:
        for arch in ARCHS:
            R.train(arch, s)
    suffix = "_no_pretrained" if R.a.offline else ""
    for s in seeds:
        for greedy in (False, True):
            nm = f"{'greedy' if greedy else 'all6'}_s{s}"
            r = py("src.research.ensemble", "--runs_dir", R.a.runs_dir, "--pattern", f"*_group_none_s{s}{suffix}",
                   "--name", nm, *(["--greedy"] if greedy else []))
            print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-800:])
    s0 = [str(Path(R.a.runs_dir, name_of(a, seeds[0], extra=["--no_pretrained"] if R.a.offline else [])))
          for a in ARCHS]
    s0 = [p for p in s0 if Path(p, "best.pth").exists()]
    if s0:
        r = py("src.explain.gradcam", "--runs", *s0, "--data_root", R.a.root, "--runs_dir", R.a.runs_dir,
               "--n_audit", "40" if R.a.smoke else "400", "--out", R.a.results)
        print("\n".join(r.stdout.strip().splitlines()[-8:]) or r.stderr[-1500:])


def stage_audits(R, seeds):
    for s in seeds:
        for arch in AUDIT_ARCHS:
            R.train(arch, s, split="image")
            for m in MASKS:
                R.train(arch, s, mask=m)
        for flag in ABLATIONS:
            R.train("efficientnet_b0", s, extra=[flag])


def stage_analysis(R):
    for d in glob.glob(R.a.input_glob, recursive=True):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metrics.json")):
            dst = Path(R.a.runs_dir, os.path.basename(d))
            if not dst.exists():
                shutil.copytree(d, dst, ignore=shutil.ignore_patterns("*.pth"))
    ab = "no_pretrained" if R.a.offline else "none"
    for mod in ("src.research.aggregate", "src.research.plots"):
        r = py(mod, "--runs_dir", R.a.runs_dir, "--main_ablation", ab,
               *(["--out", R.a.results] if mod.endswith("aggregate") else ["--results", R.a.results]))
        print(r.stdout[-4000:] if r.returncode == 0 else r.stderr[-2500:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "main", "audits", "analysis", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--data_root")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--hours", type=float, default=0, help="stop starting new runs after this many hours")
    ap.add_argument("--input_glob", default="/kaggle/input/**/runs/*")
    ap.add_argument("--offline", action="store_true", help="local tests only: no ImageNet download")
    a = ap.parse_args(argv)
    a.smoke = a.stage == "smoke"
    if a.smoke:
        a.epochs, a.seeds = 1, a.seeds[:1]
    a.root = find_root(a.data_root)
    print(f"data: {a.root}\nstage: {a.stage} | seeds {a.seeds} | epochs {a.epochs}")
    R = Runner(a)
    if a.stage in ("smoke", "main", "all"):
        stage_main(R, a.seeds)
    if a.stage in ("smoke", "audits", "all"):
        stage_audits(R, a.seeds)
    if a.stage in ("smoke", "analysis", "all"):
        stage_analysis(R)
    summary = {"stage": a.stage, "hours": round((time.time() - T0) / 3600, 2), "done": len(R.done),
               "skipped_already_done": len(R.skipped), "failed": R.failed}
    Path(a.results).mkdir(parents=True, exist_ok=True)
    Path(a.results, f"run_summary_{a.stage}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
