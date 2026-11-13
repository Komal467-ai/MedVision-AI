"""Synthetic 4-class dataset in the exact Kaggle layout (images/ + masks/), so tests run offline on CPU."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASS_DIRS = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def fake_cxr(rng, k, size=96):
    img = rng.normal(90, 15, (size, size)).clip(0, 255)
    mask = np.zeros((size, size), np.uint8)
    for cx in (size // 3, 2 * size // 3):
        cv2.ellipse(mask, (cx, size // 2), (size // 8, size // 3), 0, 0, 360, 255, -1)
    img[mask > 0] = 45
    # each class gets a different pattern inside the lungs
    if k == 0:
        img[(mask > 0) & (rng.random((size, size)) < 0.3)] = 170          # patchy (COVID)
    elif k == 1:
        cv2.circle(img, (size // 3, size // 2), size // 9, 190, -1)       # one opacity
    elif k == 3:
        cv2.circle(img, (2 * size // 3, size // 2 + 8), size // 12, 150, -1)
    return img.astype(np.uint8), mask


@pytest.fixture(scope="session")
def cxr4_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("kaggle") / "COVID-19_Radiography_Dataset"
    rng = np.random.default_rng(0)
    n_per = {"COVID": 24, "Lung_Opacity": 24, "Normal": 32, "Viral Pneumonia": 20}
    for k, c in enumerate(CLASS_DIRS):
        (root / c / "images").mkdir(parents=True)
        (root / c / "masks").mkdir(parents=True)
        for i in range(n_per[c]):
            img, mask = fake_cxr(rng, k)
            name = f"{c}-{i + 1}.png"
            cv2.imwrite(str(root / c / "images" / name), img)
            cv2.imwrite(str(root / c / "masks" / name), cv2.resize(mask, (80, 80)))
        # plant exact and mirrored duplicates to test the duplicate detector
        src = cv2.imread(str(root / c / "images" / f"{c}-1.png"), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(root / c / "images" / f"{c}-dup.png"), src)
        cv2.imwrite(str(root / c / "images" / f"{c}-mirror.png"), cv2.flip(src, 1))
        for nm in (f"{c}-dup.png", f"{c}-mirror.png"):
            cv2.imwrite(str(root / c / "masks" / nm), cv2.resize(cv2.imread(
                str(root / c / "masks" / f"{c}-1.png"), cv2.IMREAD_GRAYSCALE), (80, 80)))
    return root.parent
