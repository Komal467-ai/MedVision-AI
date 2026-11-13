"""Tests: data pipeline, model zoo, Grad-CAM, metrics, training -> ensemble -> aggregate -> plots, inference.
Run: pytest -q   (offline, CPU; models built with pretrained=False)"""
import base64
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.cxrdata.dataset import (CLASSES, add_duplicate_groups, apply_mask_mode, build_index, duplicate_report,
                              leakage_count, load_image, make_loaders, make_split, prepare_table)
from src.models.zoo import ARCHS, Classifier, load_checkpoint, save_checkpoint
from src.research.metrics import bootstrap_ci, ece, fit_temperature, selective_accuracy, softmax, summary


# ------------------------------------------------------------------ data
def test_index_finds_all_classes_and_masks(cxr4_root):
    df = build_index(cxr4_root)
    assert set(df.label) == set(CLASSES)
    assert df.mask_path.notna().all() and len(df) == 24 + 24 + 32 + 20 + 8


def test_duplicate_groups_catch_exact_and_mirrored(cxr4_root):
    df = add_duplicate_groups(build_index(cxr4_root))
    for c in CLASSES:
        sub = df[df.label == c].set_index(df[df.label == c].path.map(lambda p: Path(p).name))
        g = sub.loc[f"{c}-1.png", "group"]
        assert sub.loc[f"{c}-dup.png", "group"] == g and sub.loc[f"{c}-mirror.png", "group"] == g
    assert duplicate_report(df)["images_in_duplicate_groups"] >= 12


def test_group_split_has_no_leak_image_split_can(cxr4_root):
    df = add_duplicate_groups(build_index(cxr4_root))
    g = make_split(df, "group", seed=1)
    assert leakage_count(g) == 0
    assert set(g.split) == {"train", "val", "test"}
    for s in ("train", "val", "test"):
        assert g[g.split == s].label.nunique() == 4          # stratified
    leaks = [leakage_count(make_split(df, "image", seed=s)) for s in range(10)]
    assert max(leaks) > 0                                     # the mistake we want to measure


@pytest.mark.parametrize("mode", ["none", "lungs_only", "lungs_removed", "lung_crop"])
def test_mask_modes(cxr4_root, mode):
    r = build_index(cxr4_root).iloc[0]
    img = load_image(r.path, r.mask_path, mode)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    if mode == "lungs_removed":
        m = cv2.resize(cv2.imread(r.mask_path, 0), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 127
        assert img[..., 0][m].max() == 0


def test_loaders_return_index(cxr4_root):
    df = prepare_table(str(cxr4_root), cache_dir=str(cxr4_root / "cache"))
    loaders, cw = make_loaders(df, img_size=64, batch_size=8, workers=0)
    x, y, i = next(iter(loaders["train"]))
    assert x.shape == (8, 3, 64, 64) and y.dtype == torch.long and i.shape == (8,)
    assert cw.shape == (4,) and cw[3] > cw[2]                 # rare class gets more weight


# ------------------------------------------------------------------ models + Grad-CAM
@pytest.mark.parametrize("arch", ARCHS)
def test_every_arch_forward_and_gradcam(arch):
    from src.explain.gradcam import GradCAM
    torch.manual_seed(0)
    m = Classifier(arch, 4, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    assert m(x).shape == (1, 4)
    cam, c, p = GradCAM(m)(x)
    assert cam.shape == (224, 224) and 0 <= cam.min() and cam.max() <= 1 and c in range(4)
    assert abs(p.sum() - 1) < 1e-4
    assert len(m.cam_layer._forward_hooks) == 0


def test_checkpoint_roundtrip(tmp_path):
    m = Classifier("efficientnet_b0", 4, pretrained=False).eval()
    save_checkpoint(m, tmp_path / "x.pth")
    m2, ck = load_checkpoint(tmp_path / "x.pth")
    x = torch.randn(2, 3, 224, 224)
    assert torch.allclose(m(x), m2(x), atol=1e-5) and ck["classes"] == CLASSES


def test_lung_scores():
    from src.explain.gradcam import lung_scores
    mask = np.zeros((10, 10), np.uint8); mask[:, :5] = 255
    cam = np.zeros((10, 10)); cam[2, 2] = 1
    s = lung_scores(cam, mask)
    assert s["lung_focus"] == pytest.approx(1.0) and s["pointing"] == 1.0 and s["lung_area"] == 0.5


# ------------------------------------------------------------------ metrics
def test_metrics_and_calibration():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    logits = np.eye(4)[y] * 6 + rng.normal(0, 1, (400, 4))
    m = summary(y, softmax(logits))
    assert m["accuracy"] > 0.9 and set(m["per_class"]) == set(CLASSES) and len(m["confusion_matrix"]) == 4
    T = fit_temperature(logits, y)
    assert T > 0 and ece(y, softmax(logits, T)) <= ece(y, softmax(logits)) + 0.02
    lo, hi = bootstrap_ci(y, softmax(logits), "macro_f1", n_boot=100)
    assert lo <= m["macro_f1"] <= hi
    sa = selective_accuracy(y, softmax(logits))
    assert sa["80%"] >= sa["100%"]


# ------------------------------------------------------------------ end-to-end research pipeline
def test_end_to_end_train_ensemble_aggregate_plots(cxr4_root, tmp_path):
    from src import train
    from src.research import aggregate, ensemble, plots
    runs = tmp_path / "runs"
    common = ["--data_root", str(cxr4_root), "--epochs", "1", "--bs", "8", "--workers", "0", "--img_size", "64",
              "--no_pretrained", "--runs_dir", str(runs)]
    for arch in ("resnet50", "efficientnet_b0"):
        r = train.main(common + ["--arch", arch, "--seed", "0"])
        assert set(r["test"]) >= {"accuracy", "macro_f1", "macro_auroc", "ece"}
    train.main(common + ["--arch", "efficientnet_b0", "--seed", "0", "--split_mode", "image"])
    train.main(common + ["--arch", "efficientnet_b0", "--seed", "0", "--mask_mode", "lungs_removed"])
    ens = ensemble.main(["--pattern", "*_group_none_s0_no_pretrained", "--runs_dir", str(runs), "--name", "t"])
    assert len(ens["members"]) == 2
    out = tmp_path / "results"
    aggregate.main(["--runs_dir", str(runs), "--out", str(out), "--main_ablation", "no_pretrained"])
    for f in ("all_runs.csv", "table_main.md", "table_ci.csv", "pairwise_mcnemar.csv"):
        assert (out / f).exists()
    plots.main(["--runs_dir", str(runs), "--results", str(out), "--main_ablation", "no_pretrained"])
    for f in ("fig_model_comparison.png", "fig_leakage.png", "fig_shortcut.png", "fig_confusion_best.png",
              "fig_roc_best.png", "fig_reliability.png", "fig_per_class_sens.png"):
        assert (out / f).exists(), f


# ------------------------------------------------------------------ inference (API)
def test_inference_single_and_ensemble(tmp_path, monkeypatch, cxr4_root):
    for arch in ("efficientnet_b0", "densenet121"):
        save_checkpoint(Classifier(arch, 4, pretrained=False).eval(), tmp_path / f"{arch}.pth")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import src.inference as inf
    inf = importlib.reload(inf)
    path = next(Path(cxr4_root).rglob("images/*.png"))
    r = inf.predict_xray(path.read_bytes())
    assert r["prediction"] in CLASSES and r["ensemble"] is True and len(r["models"]) == 2
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
    img = cv2.imdecode(np.frombuffer(base64.b64decode(r["gradcam_png_base64"]), np.uint8), 1)
    assert img.shape == (224, 224, 3)
    json.dumps(r)
    with pytest.raises(ValueError):
        inf.predict_xray(b"not an image")


def test_inference_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "none"))
    import src.inference as inf
    inf = importlib.reload(inf)
    with pytest.raises(inf.ModelNotAvailable):
        inf.predict_xray(b"x")
