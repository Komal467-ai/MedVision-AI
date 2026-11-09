# M3 — ML Engineer (Diagnosis): handover

Everything M3 owns in the 8-drop plan, in the team repo layout. Copy these folders into `medical-ai/` on a branch (e.g. `m3/diagnosis`) and open PRs per drop for M1 to review.

## Deliverables checklist
| Drop | Deliverable | File | Status |
|---|---|---|---|
| 2 | PyTorch Dataset + DataLoader (patient-level split, leak check) | `src/data/dataset.py` | ✅ code + tests |
| 2 | Albumentations transforms | `src/preprocessing/augment.py` | ✅ code + tests |
| 2 | Unit tests | `tests/test_preprocessing.py` (+ `tests/test_diagnosis_models.py`) | ✅ passing |
| 3 | EfficientNet-B0 classifier | `src/diagnosis/classifier.py` | ✅ |
| 3 | Training loop (5 ep, AdamW, CE) | `src/diagnosis/train_classifier.py` | ✅ |
| 3 | Metrics + confusion matrix + ROC | `src/diagnosis/evaluate.py` | ✅ |
| 3 | Grad-CAM (2 samples) | `src/diagnosis/gradcam.py` | ✅ |
| 3 | Notebook | `notebooks/02_cnn_training.ipynb` | ✅ Kaggle-ready |
| 4 | U-Net with skip connections | `src/diagnosis/unet.py` | ✅ |
| 4 | Dice + BCE loss, Dice/IoU metrics | `src/diagnosis/losses.py` | ✅ |
| 4 | Training loop | `src/diagnosis/train_unet.py` | ✅ |
| 4 | Test Dice/IoU + overlays | `src/diagnosis/evaluate_unet.py` | ✅ |
| 4 | Notebook | `notebooks/03_unet_training.ipynb` | ✅ Kaggle-ready |
| 5–6 | CNN features + LSTM for M4 | `src/shared/cnn_features.py`, `src/shared/phase_lstm.py` | ✅ |
| 7 | API inference functions | `src/diagnosis/inference.py` | ✅ |
| 8 | Report §2 + §3 (diagnosis) | `docs/report_M3_sections.md` | ✅ draft, results to fill |
| 8 | Viva prep | `docs/viva_M3.md` | ✅ |
| — | `best_cnn.pth` (≥ 85 % acc), `best_unet.pth` (Dice ≥ 0.80) | `models/` | ⏳ produced by running the notebooks on Kaggle GPU |
| — | `results/*.png` | `results/` | ⏳ produced by the notebooks |

## Run order on Kaggle (GPU on)
```bash
# classification  (~30 min)
python -m src.diagnosis.train_classifier --data_root <.../chest_xray> --epochs 5
python -m src.diagnosis.evaluate        --ckpt models/best_cnn.pth --data_root <...> --tune_threshold
python -m src.diagnosis.gradcam         --ckpt models/best_cnn.pth --data_root <...>
# segmentation    (~20–30 min)
python -m src.diagnosis.train_unet      --img_dir <images> --mask_dir <masks> --epochs 25
python -m src.diagnosis.evaluate_unet   --ckpt models/best_unet.pth --img_dir <images> --mask_dir <masks>
```
Use the same `--seed` (default 42) for train and evaluate so the test patients are identical.
With M2's split CSV use `--csv data/splits/cxr_split.csv` (columns `path,label,patient_id,split`).

## Contract for M5 (FastAPI) — Drop 5/7
```python
from src.diagnosis.inference import predict_classification, predict_segmentation, ModelNotAvailable, models_status

@app.post("/predict/diagnosis")
async def diagnosis(file: UploadFile = File(...)):
    try:
        return predict_classification(await file.read())       # bytes, path or RGB array all work
    except ModelNotAvailable as e: raise HTTPException(503, str(e))
    except ValueError as e:        raise HTTPException(400, str(e))

@app.post("/predict/segmentation")
async def segmentation(file: UploadFile = File(...)):
    ...predict_segmentation(await file.read())...
```
`predict_classification` → `{task, model, prediction, confidence, probabilities{NORMAL,PNEUMONIA}, threshold, gradcam_png_base64, inference_ms, disclaimer}`
`predict_segmentation` → `{task, model, tumour_detected, tumour_area_percent, tumour_pixels, n_regions, bbox, mean_confidence_in_mask, mask_png_base64, overlay_png_base64, image_size, inference_ms, disclaimer}`
Streamlit: `st.image(base64.b64decode(r["gradcam_png_base64"]))`.

## Docker notes (M1/M5)
* Mount weights: `volumes: ["./models:/app/models"]`, env `MODEL_DIR=/app/models` (optional `DEVICE=cpu`).
* No internet needed at runtime — networks are built with `pretrained=False` before loading checkpoints.
* Use `opencv-python-headless` in the image (no GUI libs).
* Weights are **not** committed to git — M2 keeps them on Drive (Drop 6).

## requirements.txt additions (M1)
```
torch>=2.1
torchvision>=0.16
albumentations>=1.4
opencv-python-headless
scikit-learn
pandas
numpy
matplotlib
pytest
```

## Tests
`pytest -q tests/` — fully offline, CPU, synthetic data (builds a fake Kaggle-style folder tree and fake MRI slices).
