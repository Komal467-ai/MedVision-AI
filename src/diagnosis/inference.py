"""Inference functions the API calls (M3, Drop 7).

M5 (FastAPI) uses exactly these two functions:

    from src.diagnosis.inference import predict_classification, predict_segmentation
    predict_classification(image)   # -> dict (JSON-serialisable)
    predict_segmentation(image)     # -> dict (JSON-serialisable)

`image` can be a file path, raw bytes (UploadFile.read()), or an RGB numpy array.

Docker notes (the usual model-loading problems, fixed here):
  * weights folder comes from env MODEL_DIR (default ./models) - mount it as a volume
  * models are built with pretrained=False before loading the checkpoint, so the container
    never tries to download ImageNet weights (no internet needed at run time)
  * map_location -> CPU works in a GPU-less container; DEVICE env var can force cuda/cpu
  * models are loaded once, cached, and protected by a lock (uvicorn uses threads)
"""
from __future__ import annotations

import base64
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.preprocessing.augment import classification_transforms, segmentation_transforms

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
CNN_FILE = os.getenv("CNN_WEIGHTS", "best_cnn.pth")
UNET_FILE = os.getenv("UNET_WEIGHTS", "best_unet.pth")
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DISCLAIMER = "Research prototype for decision support only - not a medical diagnosis."

_cache: dict = {}
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


class ModelNotAvailable(RuntimeError):
    """Raised when weights are missing -> API should return HTTP 503."""


# --------------------------------------------------------------------------- helpers
def _weights(fname: str) -> Path:
    p = MODEL_DIR / fname
    if not p.exists():
        raise ModelNotAvailable(f"Weights not found: {p.resolve()} (set MODEL_DIR or train the model).")
    return p


def _get(key: str):
    with _load_lock:
        if key not in _cache:
            if key == "cnn":
                from src.diagnosis.classifier import load_classifier
                _cache[key] = load_classifier(_weights(CNN_FILE), DEVICE)
            elif key == "unet":
                from src.diagnosis.unet import load_unet
                _cache[key] = load_unet(_weights(UNET_FILE), DEVICE)
        return _cache[key]


def _read_gray(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        g = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, (bytes, bytearray)):
        g = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, np.ndarray):
        g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported image input type: {type(image)}")
    if g is None:
        raise ValueError("Could not decode image - upload a valid PNG/JPG.")
    return g


def _read_rgb(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    elif isinstance(image, (bytes, bytearray)):
        bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image input type: {type(image)}")
    if bgr is None:
        raise ValueError("Could not decode image - upload a valid PNG/JPG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb)
    return base64.b64encode(buf.tobytes()).decode()


def models_status() -> dict:
    return {"cnn": (MODEL_DIR / CNN_FILE).exists(), "unet": (MODEL_DIR / UNET_FILE).exists(),
            "model_dir": str(MODEL_DIR.resolve()), "device": DEVICE}


# --------------------------------------------------------------------------- public API
def predict_classification(image, with_gradcam: bool = True, threshold: float | None = None) -> dict:
    """Chest X-ray -> NORMAL / PNEUMONIA + confidence (+ Grad-CAM overlay as base64 PNG)."""
    from src.diagnosis.gradcam import GradCAM, overlay_cam

    t0 = time.time()
    model, ck = _get("cnn")
    classes, size = ck["classes"], ck.get("img_size", 224)
    thr = threshold if threshold is not None else float(ck.get("val_metrics", {}).get("threshold", 0.5))

    gray = _read_gray(image)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)  # same as training
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    x = classification_transforms(train=False, size=size)(image=rgb)["image"].unsqueeze(0).to(DEVICE)

    with _infer_lock:
        if with_gradcam:
            cam, _, probs = GradCAM(model, model.gradcam_layer)(x, class_idx=None)
        else:
            with torch.no_grad():
                probs = torch.softmax(model(x), 1)[0].cpu().numpy()
    p_pos = float(probs[1])
    idx = int(p_pos >= thr)
    out = {
        "task": "classification",
        "model": "EfficientNet-B0",
        "prediction": classes[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "threshold": thr,
        "gradcam_png_base64": None,
        "inference_ms": None,
        "disclaimer": DISCLAIMER,
    }
    if with_gradcam:
        # Grad-CAM always explains the PREDICTED class shown to the user
        if int(np.argmax(probs)) != idx:
            with _infer_lock:
                cam, _, _ = GradCAM(model, model.gradcam_layer)(x, class_idx=idx)
        small = cv2.resize(rgb, (x.shape[-1], x.shape[-2])) / 255.0
        out["gradcam_png_base64"] = _png_b64(overlay_cam(small, cam))
    out["inference_ms"] = round((time.time() - t0) * 1000, 1)
    return out


def predict_segmentation(image, threshold: float = 0.5) -> dict:
    """MRI slice -> binary tumour mask, overlay, tumour area % and bounding box."""
    t0 = time.time()
    model, ck = _get("unet")
    size = ck.get("img_size", 256)
    rgb = _read_rgb(image)
    h, w = rgb.shape[:2]
    x = segmentation_transforms(train=False, size=size)(image=rgb, mask=np.zeros((h, w), np.float32))["image"]
    with _infer_lock, torch.no_grad():
        prob = torch.sigmoid(model(x.unsqueeze(0).to(DEVICE)))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)  # back to original resolution
    mask = (prob > threshold).astype(np.uint8)

    overlay = rgb.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
    bbox = None
    if mask.any():
        ys, xs = np.where(mask)
        bbox = {"x_min": int(xs.min()), "y_min": int(ys.min()), "x_max": int(xs.max()), "y_max": int(ys.max())}
    return {
        "task": "segmentation",
        "model": "U-Net",
        "tumour_detected": bool(mask.any()),
        "tumour_area_percent": round(float(mask.mean()) * 100, 3),
        "tumour_pixels": int(mask.sum()),
        "n_regions": len(contours),
        "bbox": bbox,
        "mean_confidence_in_mask": round(float(prob[mask > 0].mean()), 4) if mask.any() else None,
        "mask_png_base64": _png_b64(mask * 255),
        "overlay_png_base64": _png_b64(overlay),
        "image_size": [h, w],
        "inference_ms": round((time.time() - t0) * 1000, 1),
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":  # quick manual test:  python -m src.diagnosis.inference cls path/to/xray.jpeg
    import json
    import sys

    kind, path = sys.argv[1], sys.argv[2]
    r = predict_classification(path) if kind == "cls" else predict_segmentation(path)
    print(json.dumps({k: (v[:40] + "...") if isinstance(v, str) and len(v) > 60 else v for k, v in r.items()}, indent=2))
