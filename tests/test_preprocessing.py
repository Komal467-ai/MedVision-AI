"""Unit tests for M3 Drop 2: dataset.py + augment.py.   Run:  pytest -q tests/test_preprocessing.py"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (CLASSES, ChestXrayDataset, SegmentationDataset, assert_no_patient_leak,
                              build_cxr_index, class_weights, find_seg_pairs, load_cxr_table, load_xray,
                              make_cls_loaders, make_seg_loaders, patient_id_from_filename,
                              patient_level_split)
from src.preprocessing.augment import (classification_transforms, denormalize,
                                       segmentation_transforms)


# ---------------------------------------------------------------- patient ids
@pytest.mark.parametrize("name,pid", [
    ("person1_bacteria_1.jpeg", "person1"),
    ("person1000_virus_1681.jpeg", "person1000"),
    ("IM-0115-0001.jpeg", "IM-0115"),
    ("NORMAL2-IM-1427-0001.jpeg", "NORMAL2-IM-1427"),
    ("random_name.png", "random_name"),
])
def test_patient_id_from_filename(name, pid):
    assert patient_id_from_filename(name) == pid


# ---------------------------------------------------------------- transforms
def test_classification_transform_shape_and_norm():
    img = np.random.randint(0, 255, (300, 260, 3), np.uint8)
    for train in (True, False):
        x = classification_transforms(train)(image=img)["image"]
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32
        assert -3 < x.min() < x.max() < 3  # ImageNet-normalised range


def test_eval_transform_is_deterministic():
    img = np.random.randint(0, 255, (128, 128, 3), np.uint8)
    tf = classification_transforms(train=False)
    assert torch.equal(tf(image=img)["image"], tf(image=img)["image"])


def test_segmentation_transform_keeps_mask_aligned():
    mask = np.zeros((128, 128), np.float32)
    mask[20:60, 70:110] = 1
    img = np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    tf = segmentation_transforms(train=True)
    for _ in range(10):  # random flips/rotations must move image and mask together
        out = tf(image=img, mask=mask)
        x, m = out["image"], out["mask"]
        assert x.shape == (3, 256, 256) and m.shape == (256, 256)
        bright = denormalize(x)[..., 0] > 0.5
        mm = m.numpy() > 0.5
        iou = (bright & mm).sum() / max((bright | mm).sum(), 1)
        assert iou > 0.85


def test_denormalize_roundtrip():
    img = np.random.randint(0, 255, (224, 224, 3), np.uint8)
    x = classification_transforms(train=False)(image=img)["image"]
    assert np.abs(denormalize(x) - img / 255.0).max() < 0.01


# ---------------------------------------------------------------- classification data
def test_load_xray_is_rgb_uint8(cxr_root):
    p = next(cxr_root.rglob("*.jpeg"))
    img = load_xray(p)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    assert np.array_equal(img[..., 0], img[..., 1])


def test_index_and_patient_split_no_leak(cxr_root):
    df = build_cxr_index(cxr_root)
    assert len(df) == 120 and set(df["label"]) == set(CLASSES)
    sp = patient_level_split(df, seed=0)
    assert set(sp["split"]) == {"train", "val", "test"}
    assert_no_patient_leak(sp)
    frac = sp["split"].value_counts(normalize=True)
    assert 0.6 < frac["train"] < 0.8
    for s in ("train", "val", "test"):  # both classes present in every split
        assert sp[sp["split"] == s]["label"].nunique() == 2


def test_leak_detector_catches_leak():
    df = pd.DataFrame({"patient_id": ["a", "a", "b"], "split": ["train", "test", "val"]})
    with pytest.raises(AssertionError):
        assert_no_patient_leak(df)


def test_m2_csv_input(cxr_root, tmp_path):
    df = patient_level_split(build_cxr_index(cxr_root))
    df["label"] = df["label"].map(CLASSES.index)  # M2 might write 0/1 labels
    csv = tmp_path / "split.csv"
    df.to_csv(csv, index=False)
    back = load_cxr_table(csv=str(csv))
    assert set(back["label"]) == set(CLASSES)


def test_dataset_item(cxr_root):
    df = build_cxr_index(cxr_root)
    ds = ChestXrayDataset(df, classification_transforms(False))
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and y.dtype == torch.long and int(y) in (0, 1)


def test_class_weights_inverse_frequency():
    w = class_weights([0, 1, 1, 1])
    assert w[0] > w[1] and torch.isclose((w * torch.tensor([1., 3.])).sum(), torch.tensor(4.))


def test_two_batches_load(cxr_root):
    """Drop-2 acceptance check: 2 batches load correctly."""
    loaders, w, df = make_cls_loaders(str(cxr_root), batch_size=8, num_workers=0)
    it = iter(loaders["train"])
    for _ in range(2):
        x, y = next(it)
        assert x.shape == (8, 3, 224, 224) and y.shape == (8,)
        assert torch.isfinite(x).all()
    assert w.shape == (2,)


# ---------------------------------------------------------------- segmentation data
def test_seg_pairs_layout_a(seg_dirs):
    df = find_seg_pairs(img_dir=seg_dirs / "images", mask_dir=seg_dirs / "masks")
    assert len(df) == 48 and df["patient_id"].nunique() == 12
    x, m = SegmentationDataset(df)[1]
    assert x.shape == (3, 256, 256) and m.shape == (1, 256, 256)
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}


def test_seg_pairs_layout_b(lgg_root):
    df = find_seg_pairs(root=lgg_root)
    assert len(df) == 18 and df["patient_id"].nunique() == 6


def test_seg_loaders_patient_split(seg_dirs):
    loaders, df = make_seg_loaders(img_dir=seg_dirs / "images", mask_dir=seg_dirs / "masks",
                                   batch_size=4, num_workers=0)
    assert_no_patient_leak(df)
    x, m = next(iter(loaders["train"]))
    assert x.shape == (4, 3, 256, 256) and m.shape == (4, 1, 256, 256)
