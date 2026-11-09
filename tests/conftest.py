"""Synthetic fixtures so the tests run anywhere (no dataset, no GPU, no internet)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _fake_xray(rng, pneumonia: bool, size=96):
    img = (rng.normal(90, 20, (size, size))).clip(0, 255)
    cv2.ellipse(img, (size // 3, size // 2), (size // 7, size // 3), 0, 0, 360, 40, -1)      # lungs
    cv2.ellipse(img, (2 * size // 3, size // 2), (size // 7, size // 3), 0, 0, 360, 40, -1)
    if pneumonia:  # bright opacity
        cv2.circle(img, (size // 3, size // 2 + 10), size // 10, 200, -1)
    return img.astype(np.uint8)


@pytest.fixture(scope="session")
def cxr_root(tmp_path_factory):
    """Mimics Kaggle layout <root>/{train,test,val}/{NORMAL,PNEUMONIA}/ with Kaggle file names."""
    root = tmp_path_factory.mktemp("chest_xray")
    rng = np.random.default_rng(0)
    for split in ("train", "test", "val"):
        for cls in ("NORMAL", "PNEUMONIA"):
            (root / split / cls).mkdir(parents=True)
    k = 0
    for pid in range(30):  # 30 normal patients, 2 images each
        for j in range(2):
            split = ["train", "test", "val"][pid % 3]
            cv2.imwrite(str(root / split / "NORMAL" / f"IM-{pid:04d}-{j+1:04d}.jpeg"), _fake_xray(rng, False))
    for pid in range(30):  # 30 pneumonia patients, 2 images each
        for j in range(2):
            k += 1
            split = ["train", "test", "val"][pid % 3]
            kind = "bacteria" if pid % 2 else "virus"
            cv2.imwrite(str(root / split / "PNEUMONIA" / f"person{pid}_{kind}_{k}.jpeg"), _fake_xray(rng, True))
    return root


@pytest.fixture(scope="session")
def seg_dirs(tmp_path_factory):
    """Layout A: images/ + masks/ with names <case>_<slice>.png (like M2's BraTS slices)."""
    base = tmp_path_factory.mktemp("seg")
    (base / "images").mkdir(); (base / "masks").mkdir()
    rng = np.random.default_rng(1)
    for case in range(12):
        for z in range(4):
            img = rng.normal(60, 10, (64, 64)).clip(0, 255).astype(np.uint8)
            mask = np.zeros((64, 64), np.uint8)
            if z > 0:
                c = tuple(int(v) for v in rng.integers(18, 46, 2))
                cv2.circle(mask, c, int(rng.integers(5, 12)), 255, -1)
                img[mask > 0] = 200
            name = f"BraTS_{case:03d}_{z:03d}.png"
            cv2.imwrite(str(base / "images" / name), img)
            cv2.imwrite(str(base / "masks" / name), mask)
    return base


@pytest.fixture(scope="session")
def lgg_root(tmp_path_factory):
    """Layout B: Kaggle LGG-MRI style <root>/<patient>/<name>.tif + <name>_mask.tif"""
    root = tmp_path_factory.mktemp("kaggle_3m")
    rng = np.random.default_rng(2)
    for p in range(6):
        d = root / f"TCGA_XX_{p:04d}_20000101"
        d.mkdir()
        for z in range(3):
            img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), np.uint8)
            cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)
            cv2.imwrite(str(d / f"{d.name}_{z+1}.tif"), img)
            cv2.imwrite(str(d / f"{d.name}_{z+1}_mask.tif"), mask)
    return root
