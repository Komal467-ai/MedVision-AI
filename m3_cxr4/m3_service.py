"""Import M3's chest X-ray model from anywhere in the team repo.

The team repo and this module both have a package called `src`, so a plain import would clash.
This loads m3_cxr4/src under the hood and gives the team's `src` back afterwards.

    from m3_cxr4.m3_service import predict_xray, models_status, ModelNotAvailable
    result = predict_xray(image_bytes)   # prediction, confidence, probabilities, Grad-CAM (base64)
"""
import importlib
import os
import sys
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
os.environ.setdefault("MODEL_DIR", str(Path(HERE, "models")))   # put best.pth file(s) here


def _load():
    team = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in team:
        del sys.modules[k]
    sys.path.insert(0, HERE)
    try:
        inf = importlib.import_module("src.inference")
        importlib.import_module("src.models.zoo")          # used lazily by predict_xray
    finally:
        sys.path.remove(HERE)
        m3_pkg = sys.modules.pop("src")                      # keep src.cxrdata/models/... (names the team repo doesn't use)
        sys.modules.update(team)                             # give the team's own `src` back
        if "src" not in sys.modules:
            sys.modules["_m3_src"] = m3_pkg
    return inf


_inf = _load()
predict_xray = _inf.predict_xray
models_status = _inf.models_status
ModelNotAvailable = _inf.ModelNotAvailable
