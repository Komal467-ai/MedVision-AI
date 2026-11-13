"""Data pipeline for 4-class chest X-ray classification.

Dataset: COVID-19 Radiography Database (Kaggle: tawsifurrahman/covid19-radiography-database)
    21,165 PNG chest X-rays, 299x299, four classes, each with a lung mask (256x256):
      COVID 3,616 | Lung_Opacity 6,012 | Normal 10,192 | Viral Pneumonia 1,345
    Citations: Chowdhury et al., IEEE Access 8 (2020) 132665-132676;
               Rahman et al., Computers in Biology and Medicine 132 (2021) 104319.

Expected layout (as downloaded):
    <root>/COVID-19_Radiography_Dataset/<Class>/images/*.png
    <root>/COVID-19_Radiography_Dataset/<Class>/masks/*.png       (same file names)
Older versions without images/ masks/ sub-folders are also accepted (no masks then).

Research-specific choices (see docs/RESEARCH_PLAN.md):
  * The database has NO patient IDs. To stop the same X-ray (or a mirrored / re-saved copy)
    leaking between train and test, near-duplicates are found with a perceptual hash and
    every duplicate group is kept inside one split ("group-aware split").
  * --split_mode image ignores the groups on purpose, to measure how much leakage inflates scores.
  * --mask_mode uses the lung masks: lungs_only / lungs_removed / lung_crop, for the shortcut audit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.cxrdata.augment import build_transforms

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
SHORT = ["COVID", "Opacity", "Normal", "Viral"]
IMG_EXTS = {".png", ".jpg", ".jpeg"}
MASK_MODES = ("none", "lungs_only", "lungs_removed", "lung_crop")


# ============================================================================ indexing
def _class_of(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        for c in CLASSES:
            if part.lower() == c.lower():
                return c
    return None


def build_index(root: str | Path) -> pd.DataFrame:
    """Scan the dataset -> DataFrame[path, mask_path, label, y]. Masks are optional."""
    rows = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in IMG_EXTS or "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        if p.parent.name.lower() == "masks":
            continue
        c = _class_of(p)
        if c is None:
            continue
        mask = None
        if p.parent.name.lower() == "images":
            m = p.parent.parent / "masks" / p.name
            mask = str(m) if m.exists() else None
        rows.append({"path": str(p), "mask_path": mask, "label": c, "y": CLASSES.index(c)})
    if not rows:
        raise FileNotFoundError(f"No images of classes {CLASSES} under {root}")
    df = pd.DataFrame(rows)
    # some Kaggle mirrors contain the dataset twice; keep one copy per class/file name
    df["key"] = df["label"] + "/" + df["path"].map(lambda s: Path(s).name)
    df = df.drop_duplicates("key").drop(columns="key").sort_values("path").reset_index(drop=True)
    return df


# ============================================================================ near-duplicates
def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash: 64-bit fingerprint robust to resizing, compression and brightness."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def add_duplicate_groups(df: pd.DataFrame, cache: str | None = None) -> pd.DataFrame:
    """Adds columns md5, phash, group. Images with the same MD5, the same dHash, or the same
    dHash after a horizontal flip are put in one group (union-find)."""
    if cache and Path(cache).exists():
        c = pd.read_csv(cache)
        if len(c) == len(df) and (c["path"].values == df["path"].values).all():
            return c
    md5s, hashes, flips = [], [], []
    for p in df["path"]:
        raw = Path(p).read_bytes()
        md5s.append(hashlib.md5(raw).hexdigest())
        g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        hashes.append(dhash(g))
        flips.append(dhash(cv2.flip(g, 1)))
    df = df.copy()
    df["md5"], df["phash"] = md5s, [f"{h:016x}" for h in hashes]

    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    first: dict = {}
    for i, keys in enumerate(zip(md5s, hashes, flips)):
        for k in (("m", keys[0]), ("h", keys[1]), ("h", keys[2])):
            if k in first:
                union(i, first[k])
            else:
                first[k] = i
    df["group"] = [find(i) for i in range(len(df))]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def duplicate_report(df: pd.DataFrame) -> dict:
    sizes = df.groupby("group").size()
    conflict = df.groupby("group")["label"].nunique()
    return {"images": int(len(df)), "groups": int(len(sizes)),
            "images_in_duplicate_groups": int(sizes[sizes > 1].sum()),
            "groups_with_conflicting_labels": int((conflict > 1).sum())}


# ============================================================================ splitting
def make_split(df: pd.DataFrame, mode: str = "group", ratios=(0.70, 0.15, 0.15), seed: int = 42,
               drop_conflicts: bool = True) -> pd.DataFrame:
    """Adds 'split'. mode='group': duplicate groups never cross splits (main protocol).
    mode='image': every image assigned independently (reproduces the common leakage mistake).
    Both are stratified by class."""
    assert mode in ("group", "image")
    df = df.copy()
    if "group" not in df:
        df["group"] = np.arange(len(df))
    if drop_conflicts:  # same picture filed under two diseases -> unusable label
        bad = df.groupby("group")["label"].nunique()
        df = df[~df["group"].isin(bad[bad > 1].index)].reset_index(drop=True)
    unit = "group" if mode == "group" else "path"
    rng = np.random.default_rng(seed)
    df["split"] = ""
    for c in CLASSES:
        sub = df[df["label"] == c]
        units = np.array(sorted(map(str, sub[unit].unique())))
        rng.shuffle(units)
        n_tr, n_va = int(round(ratios[0] * len(units))), int(round(ratios[1] * len(units)))
        assign = {u: "train" for u in units[:n_tr]}
        assign.update({u: "val" for u in units[n_tr:n_tr + n_va]})
        assign.update({u: "test" for u in units[n_tr + n_va:]})
        df.loc[sub.index, "split"] = sub[unit].astype(str).map(assign)
    return df


def leakage_count(df: pd.DataFrame) -> int:
    """Number of duplicate groups that appear in more than one split (0 for mode='group')."""
    return int((df.groupby("group")["split"].nunique() > 1).sum())


def prepare_table(data_root: str, split_mode: str = "group", seed: int = 42, cache_dir: str = "results/cache",
                  dedup: bool = True) -> pd.DataFrame:
    df = build_index(data_root)
    if dedup:
        df = add_duplicate_groups(df, cache=str(Path(cache_dir) / "duplicate_groups.csv"))
    return make_split(df, split_mode, seed=seed)


# ============================================================================ image loading
def apply_mask_mode(gray: np.ndarray, mask: np.ndarray | None, mode: str, dilate: int = 5) -> np.ndarray:
    if mode == "none":
        return gray
    if mask is None:
        raise ValueError(f"mask_mode={mode} needs lung masks, but this image has none")
    m = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    if dilate:
        m = cv2.dilate(m.astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool)
    if mode == "lungs_only":
        return np.where(m, gray, 0).astype(np.uint8)
    if mode == "lungs_removed":
        return np.where(m, 0, gray).astype(np.uint8)
    if mode == "lung_crop":
        ys, xs = np.where(m)
        if len(ys) == 0:
            return gray
        pad = int(0.05 * max(gray.shape))
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, gray.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, gray.shape[1])
        return gray[y0:y1, x0:x1]
    raise ValueError(mode)


def load_image(path: str, mask_path: str | None = None, mask_mode: str = "none", clahe: bool = True) -> np.ndarray:
    """-> HxWx3 uint8 RGB (3 identical channels for ImageNet-pretrained backbones)."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if (mask_path and mask_mode != "none") else None
    gray = apply_mask_mode(gray, mask, mask_mode)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class CXRDataset(Dataset):
    """Returns (image_tensor[3,S,S], label_long, row_index)."""

    def __init__(self, df: pd.DataFrame, transform, mask_mode: str = "none", clahe: bool = True):
        self.df = df.reset_index(drop=True)
        self.tf, self.mask_mode, self.clahe = transform, mask_mode, clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_image(r["path"], r.get("mask_path"), self.mask_mode, self.clahe)
        return self.tf(image=img)["image"], torch.tensor(int(r["y"]), dtype=torch.long), i


def class_weights(y, n: int = len(CLASSES)) -> torch.Tensor:
    counts = np.bincount(np.asarray(y), minlength=n).astype(float)
    return torch.tensor(counts.sum() / (n * np.maximum(counts, 1)), dtype=torch.float32)


def make_loaders(df: pd.DataFrame, img_size: int = 224, batch_size: int = 32, workers: int = 2,
                 mask_mode: str = "none", clahe: bool = True, augment: bool = True):
    out = {}
    for s in ("train", "val", "test"):
        part = df[df["split"] == s]
        ds = CXRDataset(part, build_transforms(train=(s == "train" and augment), size=img_size), mask_mode, clahe)
        out[s] = DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"), num_workers=workers,
                            pin_memory=torch.cuda.is_available(), drop_last=False)
    return out, class_weights(df[df["split"] == "train"]["y"])


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Index the dataset, find duplicates, print the split")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_mode", default="group", choices=["group", "image"])
    a = ap.parse_args()
    t = prepare_table(a.data_root, a.split_mode)
    print(json.dumps(duplicate_report(t), indent=2))
    print(t.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("duplicate groups crossing splits:", leakage_count(t))
