"""PyTorch Datasets + DataLoaders for the diagnosis models (M3, Drop 2).

Classification - Chest X-Ray Pneumonia (Kaggle, Kermany et al.)
    Expected layout (as downloaded):  <root>/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
    The official 'val' folder has only 16 images, so we IGNORE the official split and
    re-split all images at PATIENT level 70/15/15 (no patient in more than one split).
    If M2's src/data/split.py has already written a CSV, pass it with --csv instead:
        columns: path, label, patient_id, split      (split in {train,val,test})

Segmentation - image/mask pairs prepared by M2 (BraTS slices / LUNA16 / LGG-MRI)
    Layout A:  <img_dir>/<name>.png  +  <mask_dir>/<name>.png   (identical names)
    Layout B:  <root>/**/<name>.tif  +  <root>/**/<name>_mask.tif (Kaggle LGG-MRI layout)
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.preprocessing.augment import classification_transforms, segmentation_transforms

CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------- helpers
def patient_id_from_filename(name: str) -> str:
    """Kaggle CXR naming ->  patient id.
    person1_bacteria_1.jpeg      -> person1
    person1000_virus_1681.jpeg   -> person1000
    IM-0115-0001.jpeg            -> IM-0115
    NORMAL2-IM-1427-0001.jpeg    -> NORMAL2-IM-1427
    Anything else: file stem (each image treated as its own patient - safe default)."""
    stem = Path(name).stem
    m = re.match(r"^(person\d+)_", stem)
    if m:
        return m.group(1)
    m = re.match(r"^((?:NORMAL2-)?IM-\d+)-\d+$", stem)
    if m:
        return m.group(1)
    return stem


def load_xray(path: str | Path, use_clahe: bool = True) -> np.ndarray:
    """Read an X-ray as grayscale, apply CLAHE (same settings as M2's preprocessing),
    return HxWx3 uint8 RGB (3 identical channels for the ImageNet-pretrained CNN)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if use_clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def build_cxr_index(root: str | Path) -> pd.DataFrame:
    """Scan the Kaggle folder tree and return DataFrame[path, label, patient_id]."""
    rows = []
    for p in Path(root).rglob("*"):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        if p.suffix.lower() in IMG_EXTS and p.parent.name.upper() in CLASSES:
            rows.append({"path": str(p), "label": p.parent.name.upper(),
                         "patient_id": f"{p.parent.name.upper()}::{patient_id_from_filename(p.name)}"})
    if not rows:
        raise FileNotFoundError(f"No NORMAL/PNEUMONIA images found under {root}")
    # the Kaggle archive contains a duplicated nested copy (chest_xray/chest_xray/...)
    df = pd.DataFrame(rows)
    df["fname"] = df["path"].map(lambda s: "/".join(Path(s).parts[-3:]))  # split/class/file
    df = df.drop_duplicates("fname").drop(columns="fname").reset_index(drop=True)
    return df


def patient_level_split(df: pd.DataFrame, ratios=(0.70, 0.15, 0.15), seed: int = 42,
                        group_col: str = "patient_id", strat_col: str | None = "label") -> pd.DataFrame:
    """Adds a 'split' column. Patients (not images) are shuffled and assigned, separately
    per class, so class balance is kept AND no patient appears in two splits."""
    assert abs(sum(ratios) - 1) < 1e-6
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""
    strata = df[strat_col].unique() if strat_col else [None]
    for s in strata:
        sub = df if s is None else df[df[strat_col] == s]
        pats = np.array(sorted(map(str, sub[group_col].unique())))  # sorted -> same split on every machine
        rng.shuffle(pats)
        n_tr = int(round(ratios[0] * len(pats)))
        n_va = int(round(ratios[1] * len(pats)))
        assign = {p: "train" for p in pats[:n_tr]}
        assign.update({p: "val" for p in pats[n_tr:n_tr + n_va]})
        assign.update({p: "test" for p in pats[n_tr + n_va:]})
        df.loc[sub.index, "split"] = sub[group_col].astype(str).map(assign)
    return df


def assert_no_patient_leak(df: pd.DataFrame, group_col: str = "patient_id") -> None:
    per_patient = df.groupby(group_col)["split"].nunique()
    leaked = per_patient[per_patient > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} patients appear in more than one split, e.g. {list(leaked.index[:5])}")


def load_cxr_table(data_root: str | None = None, csv: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Single entry point: M2's CSV if given, otherwise scan + split ourselves."""
    if csv:
        df = pd.read_csv(csv)
        need = {"path", "label", "patient_id", "split"}
        if not need.issubset(df.columns):
            raise ValueError(f"CSV must contain columns {need}, got {list(df.columns)}")
        if data_root:  # relative paths in CSV
            df["path"] = df["path"].map(lambda p: p if Path(p).is_absolute() else str(Path(data_root) / p))
        df["label"] = df["label"].map(lambda v: CLASSES[int(v)] if str(v).isdigit() else str(v).upper())
    else:
        df = patient_level_split(build_cxr_index(data_root), seed=seed)
    assert_no_patient_leak(df)
    return df


# --------------------------------------------------------------------------- classification
class ChestXrayDataset(Dataset):
    """Returns (image_tensor[3,224,224], label_long) where 0=NORMAL, 1=PNEUMONIA."""

    def __init__(self, df: pd.DataFrame, transform=None, use_clahe: bool = True):
        self.paths = df["path"].tolist()
        self.labels = [CLASSES.index(l) for l in df["label"]]
        self.tf = transform or classification_transforms(train=False)
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_xray(self.paths[i], self.use_clahe)
        x = self.tf(image=img)["image"]
        return x, torch.tensor(self.labels[i], dtype=torch.long)


def class_weights(labels, n_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss (dataset is ~3:1 PNEUMONIA:NORMAL)."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def make_cls_loaders(data_root=None, csv=None, batch_size=32, num_workers=2, seed=42,
                     balanced_sampler=False):
    df = load_cxr_table(data_root, csv, seed)
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        ds = ChestXrayDataset(part, classification_transforms(train=(split == "train")))
        sampler, shuffle = None, split == "train"
        if split == "train" and balanced_sampler:
            w = class_weights(ds.labels)[ds.labels]
            sampler, shuffle = WeightedRandomSampler(w, len(ds), replacement=True), False
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_labels = loaders["train"].dataset.labels
    return loaders, class_weights(train_labels), df


# --------------------------------------------------------------------------- segmentation
def find_seg_pairs(root=None, img_dir=None, mask_dir=None) -> pd.DataFrame:
    """Return DataFrame[image, mask, patient_id] for layout A or B (see module docstring)."""
    rows = []
    if img_dir and mask_dir:
        for p in sorted(Path(img_dir).iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            m = Path(mask_dir) / p.name
            if not m.exists():
                cand = list(Path(mask_dir).glob(p.stem + ".*"))
                if not cand:
                    continue
                m = cand[0]
            # BraTS slices from M2 are named <case>_<slice>.png -> patient = <case>
            pid = re.sub(r"_\d+$", "", p.stem)
            rows.append({"image": str(p), "mask": str(m), "patient_id": pid})
    elif root:
        for m in sorted(Path(root).rglob("*_mask.*")):
            img = m.with_name(m.name.replace("_mask", ""))
            if img.exists():
                rows.append({"image": str(img), "mask": str(m), "patient_id": m.parent.name})
    if not rows:
        raise FileNotFoundError("No image/mask pairs found - check --seg_root or --img_dir/--mask_dir")
    return pd.DataFrame(rows)


class SegmentationDataset(Dataset):
    """Returns (image[3,H,W] float, mask[1,H,W] float in {0,1})."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.imgs, self.masks = df["image"].tolist(), df["mask"].tolist()
        self.tf = transform or segmentation_transforms(train=False)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.imgs[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype("float32")
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].float().unsqueeze(0)


def make_seg_loaders(root=None, img_dir=None, mask_dir=None, batch_size=16, num_workers=2,
                     seed=42, only_positive_train=False):
    df = patient_level_split(find_seg_pairs(root, img_dir, mask_dir), seed=seed, strat_col=None)
    assert_no_patient_leak(df)
    if only_positive_train:  # optional: drop empty-mask slices from TRAIN only
        has_fg = df["mask"].map(lambda m: cv2.imread(m, cv2.IMREAD_GRAYSCALE).max() > 0)
        df = df[(df["split"] != "train") | has_fg]
    loaders = {}
    for split in ("train", "val", "test"):
        ds = SegmentationDataset(df[df["split"] == split], segmentation_transforms(train=(split == "train")))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loaders, df


if __name__ == "__main__":  # Drop 2 check: "verify 2 batches load correctly"
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root")
    ap.add_argument("--csv")
    a = ap.parse_args()
    loaders, w, df = make_cls_loaders(a.data_root, a.csv, batch_size=8, num_workers=0)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("class weights:", w.tolist())
    for b, (x, y) in enumerate(loaders["train"]):
        print(f"batch {b}: x {tuple(x.shape)} {x.dtype}  range [{x.min():.2f},{x.max():.2f}]  y {y.tolist()}")
        if b == 1:
            break
