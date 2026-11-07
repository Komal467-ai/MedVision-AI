
"""
M4 - Surgical Instrument Detection
Train YOLOv8 on the Labeled Surgical Tools dataset.
Evaluate the final model on the official test split.
"""

from pathlib import Path
import shutil
import torch
from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "surgical_instruments"
    / "yolo"
    / "data.yaml"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

BEST_MODEL_PATH = MODEL_DIR / "best_yolo.pt"
METRICS_PATH = RESULTS_DIR / "yolo_metrics.txt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. TRAINING CONFIGURATION
# --------------------------------------------------

BASE_MODEL = "yolov8n.pt"

EPOCHS = 5
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 15

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# 3. VALIDATE DATASET
# --------------------------------------------------

def validate_dataset():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    required_folders = [
        PROJECT_ROOT / "data/surgical_instruments/yolo/images/train",
        PROJECT_ROOT / "data/surgical_instruments/yolo/images/val",
        PROJECT_ROOT / "data/surgical_instruments/yolo/images/test",
        PROJECT_ROOT / "data/surgical_instruments/yolo/labels/train",
        PROJECT_ROOT / "data/surgical_instruments/yolo/labels/val",
        PROJECT_ROOT / "data/surgical_instruments/yolo/labels/test",
    ]

    for folder in required_folders:
        if not folder.exists():
            raise FileNotFoundError(
                f"Required dataset folder missing: {folder}"
            )

    print("Dataset validation successful.")


# --------------------------------------------------
# 4. TRAIN YOLOV8
# --------------------------------------------------

def train_model():

    print("\n" + "=" * 55)
    print("SURGICAL INSTRUMENT DETECTION - YOLOV8 TRAINING")
    print("=" * 55)

    print(f"Dataset: {DATA_YAML}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")

    model = YOLO(BASE_MODEL)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        workers=0,
        project=str(RESULTS_DIR),
        name="surgical_yolo_training",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        plots=True,
        save=True,
        verbose=True
    )

    # Locate the best checkpoint.
    best_checkpoint = Path(model.trainer.best)

    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_checkpoint}"
        )

    shutil.copy2(best_checkpoint, BEST_MODEL_PATH)

    print("\nTraining completed.")
    print(f"Best model saved at: {BEST_MODEL_PATH}")

    return YOLO(str(BEST_MODEL_PATH))


# --------------------------------------------------
# 5. EVALUATE ON OFFICIAL TEST DATA
# --------------------------------------------------

def evaluate_model(model):

    print("\n" + "=" * 55)
    print("EVALUATING ON OFFICIAL TEST DATA")
    print("=" * 55)

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(RESULTS_DIR),
        name="surgical_yolo_test",
        exist_ok=True
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # --------------------------------------------------
    # 6. SAVE METRICS
    # --------------------------------------------------

    report = f"""
SURGICAL INSTRUMENT DETECTION - TEST RESULTS
=============================================

Model: YOLOv8n
Dataset: Labeled Surgical Tools
Number of classes: 4

Precision:       {precision:.4f}
Recall:          {recall:.4f}
mAP@0.5:         {map50:.4f}
mAP@0.5:0.95:    {map50_95:.4f}

Target mAP@0.5:  0.70

Target achieved: {"YES" if map50 >= 0.70 else "NO"}

Best model: {BEST_MODEL_PATH}
"""

    METRICS_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"Metrics saved at: {METRICS_PATH}")


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():

    validate_dataset()

    print("\nCUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU. Training may take longer.")

    trained_model = train_model()

    evaluate_model(trained_model)

    print("\n" + "=" * 55)
    print("M4 YOLO TRAINING AND TESTING COMPLETED")
    print("=" * 55)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nTRAINING ERROR:")
        print(error)
        raise
