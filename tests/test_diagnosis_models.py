"""Tests for M3 models, losses, Grad-CAM, training scripts and inference API functions.
Run:  pytest -q tests/        (CPU, offline - models are built with pretrained=False)"""
import base64
import importlib
import json

import cv2
import numpy as np
import pytest
import torch

from src.diagnosis.classifier import build_classifier, load_classifier, save_checkpoint
from src.diagnosis.evaluate import classification_metrics
from src.diagnosis.gradcam import GradCAM, overlay_cam
from src.diagnosis.losses import DiceBCELoss, dice_iou_per_image, soft_dice
from src.diagnosis.unet import UNet, load_unet, save_unet
from src.shared.phase_lstm import FeatureWindows, PhaseLSTM, predict_video


@pytest.fixture(scope="module")
def cnn():
    torch.manual_seed(0)
    return build_classifier(2, pretrained=False).eval()


# ---------------------------------------------------------------- classifier
def test_classifier_forward_and_embed(cnn):
    x = torch.randn(2, 3, 224, 224)
    assert cnn(x).shape == (2, 2)
    assert cnn.embed(x).shape == (2, 1280)


def test_checkpoint_roundtrip(cnn, tmp_path):
    p = tmp_path / "best_cnn.pth"
    save_checkpoint(cnn, p, epoch=1, val_metrics={"auroc": 0.9})
    m2, ck = load_classifier(p)
    x = torch.randn(1, 3, 224, 224)
    assert torch.allclose(cnn(x), m2(x), atol=1e-5)
    assert ck["classes"] == ["NORMAL", "PNEUMONIA"] and ck["arch"] == "efficientnet_b0"


def test_metrics():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert m["accuracy"] == 0.75 and m["recall"] == 1.0 and m["specificity"] == 0.5
    assert m["confusion_matrix"] == [[1, 1], [0, 2]] and m["auroc"] == 1.0


# ---------------------------------------------------------------- Grad-CAM
def test_gradcam_shape_range_and_hooks_removed(cnn):
    x = torch.randn(1, 3, 224, 224)
    cam, c, probs = GradCAM(cnn, cnn.gradcam_layer)(x)
    assert cam.shape == (224, 224) and 0 <= cam.min() and cam.max() <= 1
    assert c in (0, 1) and abs(probs.sum() - 1) < 1e-5
    assert len(cnn.gradcam_layer._forward_hooks) == 0  # no hook leak between API calls
    ov = overlay_cam(np.random.rand(224, 224, 3), cam)
    assert ov.shape == (224, 224, 3) and ov.dtype == np.uint8


# ---------------------------------------------------------------- U-Net + losses
@pytest.mark.parametrize("size", [256, 250])
def test_unet_shapes(size):
    m = UNet(base=8).eval()
    assert m(torch.randn(1, 3, size, size)).shape == (1, 1, size, size)


def test_unet_default_param_count():
    n = sum(p.numel() for p in UNet().parameters())
    assert 7e6 < n < 8.5e6


def test_unet_roundtrip(tmp_path):
    m = UNet(base=8).eval()
    save_unet(m, tmp_path / "u.pth")
    m2, _ = load_unet(tmp_path / "u.pth")
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(m(x), m2(x), atol=1e-5)


def test_dice_metrics():
    t = torch.zeros(2, 1, 8, 8); t[0, :, 2:6, 2:6] = 1
    perfect = (t * 20) - 10  # logits
    d, i = dice_iou_per_image(perfect, t)
    assert torch.allclose(d, torch.ones(2)) and torch.allclose(i, torch.ones(2))  # incl. empty/empty = 1
    assert soft_dice(perfect, t)[0] > 0.99
    loss = DiceBCELoss()
    assert loss(perfect, t) < loss(torch.zeros_like(t), t)


def test_unet_can_overfit_tiny_batch():
    torch.manual_seed(0)
    m = UNet(base=8)
    x = torch.randn(2, 3, 32, 32)
    y = (x[:, :1] > 0.5).float()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    crit = DiceBCELoss()
    first = crit(m(x), y).item()
    for _ in range(60):
        opt.zero_grad(); l = crit(m(x), y); l.backward(); opt.step()
    assert l.item() < 0.5 * first


# ---------------------------------------------------------------- end-to-end smoke (training scripts)
def test_train_and_evaluate_classifier_smoke(cxr_root, tmp_path):
    from src.diagnosis import train_classifier
    hist = train_classifier.main(["--data_root", str(cxr_root), "--epochs", "1", "--bs", "8",
                                  "--workers", "0", "--no_pretrained", "--max_batches", "2",
                                  "--out", str(tmp_path / "best_cnn.pth"), "--results", str(tmp_path)])
    assert len(hist) == 1 and (tmp_path / "best_cnn.pth").exists() and (tmp_path / "training_curves.png").exists()


def test_train_unet_smoke(seg_dirs, tmp_path):
    from src.diagnosis import train_unet
    hist = train_unet.main(["--img_dir", str(seg_dirs / "images"), "--mask_dir", str(seg_dirs / "masks"),
                            "--epochs", "1", "--bs", "4", "--workers", "0", "--base", "8", "--max_batches", "2",
                            "--out", str(tmp_path / "best_unet.pth"), "--results", str(tmp_path)])
    assert len(hist) == 1 and (tmp_path / "best_unet.pth").exists()


# ---------------------------------------------------------------- inference.py (what the API calls)
@pytest.fixture()
def inference(tmp_path, monkeypatch, cnn):
    save_checkpoint(cnn, tmp_path / "best_cnn.pth")
    save_unet(UNet(base=8).eval(), tmp_path / "best_unet.pth")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import src.diagnosis.inference as inf
    return importlib.reload(inf)


def _png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_predict_classification(inference, cxr_root):
    path = next(cxr_root.rglob("*.jpeg"))
    for inp in (str(path), path.read_bytes()):
        r = inference.predict_classification(inp)
        assert r["prediction"] in ("NORMAL", "PNEUMONIA") and 0.5 <= r["confidence"] <= 1
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
        img = cv2.imdecode(np.frombuffer(base64.b64decode(r["gradcam_png_base64"]), np.uint8), 1)
        assert img.shape == (224, 224, 3)
        json.dumps(r)  # must be JSON-serialisable for FastAPI


def test_predict_segmentation(inference):
    img = np.random.randint(0, 255, (180, 200, 3), np.uint8)
    r = inference.predict_segmentation(_png_bytes(img))
    assert r["image_size"] == [180, 200] and 0 <= r["tumour_area_percent"] <= 100
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(r["mask_png_base64"]), np.uint8), 0)
    assert mask.shape == (180, 200)
    json.dumps(r)


def test_missing_weights_and_bad_image(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty"))
    import src.diagnosis.inference as inf
    inf = importlib.reload(inf)
    with pytest.raises(inf.ModelNotAvailable):
        inf.predict_classification(b"whatever")
    assert inf.models_status()["cnn"] is False


def test_bad_image_raises_value_error(inference):
    with pytest.raises(ValueError):
        inference.predict_classification(b"not an image")


# ---------------------------------------------------------------- M4 support
def test_phase_lstm():
    vids = [(np.random.randn(100, 1280).astype("float32"), np.random.randint(0, 7, 100)),
            (np.random.randn(30, 1280).astype("float32"), np.random.randint(0, 7, 30))]
    ds = FeatureWindows(vids, win=64, stride=32)
    f, y = ds[len(ds) - 1]
    assert f.shape == (64, 1280) and (y == -100).any()  # short video padded + ignored
    m = PhaseLSTM()
    logits, _ = m(torch.randn(2, 64, 1280))
    assert logits.shape == (2, 64, 7)
    assert predict_video(m, vids[0][0]).shape == (100,)
