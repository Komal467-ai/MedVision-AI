"""What the API calls: 4-class chest X-ray diagnosis with an explanation.

    from src.inference import predict_xray, models_status
    predict_xray(image_bytes_or_path)   # -> JSON-serialisable dict

MODEL_DIR (default ./models) holds one or more *.pth checkpoints produced by src.train.
One file -> single model. Several files -> soft-voting ensemble (mean probability);
Grad-CAM is drawn from the first model in alphabetical order (or CAM_MODEL=<file name>).
Built with pretrained=False and map_location -> works offline and on CPU (Docker).
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

from src.cxrdata.augment import build_transforms
from src.cxrdata.dataset import CLASSES
from src.explain.gradcam import GradCAM, overlay

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DISCLAIMER = "Research prototype for decision support only - not a medical diagnosis."
_lock = threading.Lock()
_cache: dict = {}


class ModelNotAvailable(RuntimeError):
    pass


def _models():
    with _lock:
        if "m" not in _cache:
            from src.models.zoo import load_checkpoint
            files = sorted(MODEL_DIR.glob("*.pth"))
            if not files:
                raise ModelNotAvailable(f"No .pth checkpoints in {MODEL_DIR.resolve()}")
            cam_name = os.getenv("CAM_MODEL")
            files.sort(key=lambda f: (f.name != cam_name, f.name))
            _cache["m"] = [(f.name, *load_checkpoint(f, DEVICE)) for f in files]
        return _cache["m"]


def models_status() -> dict:
    files = sorted(p.name for p in MODEL_DIR.glob("*.pth"))
    return {"model_dir": str(MODEL_DIR.resolve()), "checkpoints": files, "ensemble": len(files) > 1,
            "device": DEVICE}


def _read_gray(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        g = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, (bytes, bytearray)):
        g = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, np.ndarray):
        g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"unsupported input {type(image)}")
    if g is None:
        raise ValueError("could not decode image - upload a PNG or JPG chest X-ray")
    return g


def predict_xray(image, with_gradcam: bool = True) -> dict:
    t0 = time.time()
    members = _models()
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(_read_gray(image))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    size = members[0][2].get("img_size", 224)
    x = build_transforms(train=False, size=size)(image=rgb)["image"][None].to(DEVICE)

    probs, cam = [], None
    with _lock:
        for i, (_name, model, ck) in enumerate(members):
            if i == 0 and with_gradcam:
                cam, _, p = GradCAM(model)(x)
            else:
                with torch.no_grad():
                    p = torch.softmax(model(x).float(), 1)[0].cpu().numpy()
            probs.append(p)
    p = np.mean(probs, axis=0)
    k = int(p.argmax())
    out = {
        "task": "chest_xray_4class",
        "prediction": CLASSES[k],
        "confidence": round(float(p[k]), 4),
        "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, p)},
        "models": [f"{m[2]['arch']} ({m[0]})" for m in members],
        "ensemble": len(members) > 1,
        "gradcam_png_base64": None,
        "gradcam_from": members[0][2]["arch"] if with_gradcam else None,
        "disclaimer": DISCLAIMER,
    }
    if cam is not None:
        ov = overlay(cv2.resize(rgb, (size, size)) / 255.0, cam)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        out["gradcam_png_base64"] = base64.b64encode(buf.tobytes()).decode()
    out["inference_ms"] = round((time.time() - t0) * 1000, 1)
    return out
