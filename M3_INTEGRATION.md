# M3 — Diagnosis (chest X-ray): handover

M3's module now lives in **`m3_cxr4/`**. It is self-contained (its own `src/`, `tests/` and `requirements.txt`), so it doesn't touch the API, dashboard or M4 code.
It replaces the earlier 2-class pneumonia CNN + U-Net version.

**Task:** 4-class chest X-ray diagnosis: COVID-19, Lung Opacity, Normal, Viral Pneumonia
**Data:** COVID-19 Radiography Database (Kaggle `tawsifurrahman/covid19-radiography-database`), 21,165 images with lung masks, split 70/15/15 with near-duplicates kept in the same split
**Models:** ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, Swin-T (ImageNet-pretrained), plus soft-voting ensembles, 3 seeds each
**Checks:** Grad-CAM lung-focus audit, leakage test, shortcut test with masked lungs, calibration, ablations

## Run
```bash
cd m3_cxr4
pip install -r requirements.txt
pytest -q                                   # 20 offline tests
python -m src.run_study --stage smoke       # every step on a few batches (GPU)
```
Full study: `m3_cxr4/notebooks/kaggle_run_study.ipynb` on Kaggle (T4 GPU), run once per stage: main → audits → analysis.

## For M5 (API)
```python
from m3_cxr4.m3_service import predict_xray, ModelNotAvailable, models_status   # avoids the clash between the two `src` packages

@app.post("/predict/diagnosis")
async def predict_diagnosis(file: UploadFile = File(...)):
    try:
        return predict_xray(await file.read())   # prediction, confidence, probabilities (4 classes), gradcam_png_base64
    except ModelNotAvailable as e:
        raise HTTPException(503, str(e))
```
Put the trained `best.pth` file(s) in `m3_cxr4/models/`. They're shared on Drive, not in git; with several files, predictions use their average (ensemble).

Research prototype — not a medical device.
