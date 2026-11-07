
"""
M4 - Surgical Instrument Detection
Test the trained YOLOv8 model on a real dataset image.
"""

from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_yolo.pt"

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "surgical_instruments"
    / "yolo"
    / "images"
    / "test"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "surgical_prediction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CLASS NAMES
# --------------------------------------------------

CLASS_NAMES = {
    0: "Scalpel No. 4",
    1: "Straight Dissection Clamp",
    2: "Straight Mayo Scissor",
    3: "Curved Mayo Scissor",
}


# --------------------------------------------------
# 3. VALIDATE FILES
# --------------------------------------------------

def validate_files():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found: {TEST_IMAGE_DIR}"
        )


# --------------------------------------------------
# 4. LOAD TEST IMAGE
# --------------------------------------------------

def get_test_image():

    extensions = [".jpg", ".jpeg", ".png"]

    images = sorted(
        image
        for image in TEST_IMAGE_DIR.iterdir()
        if image.is_file()
        and image.suffix.lower() in extensions
    )

    if not images:
        raise FileNotFoundError(
            "No test images found."
        )

    return images[0]


# --------------------------------------------------
# 5. RUN DETECTION
# --------------------------------------------------

def detect_instruments():

    print("\nLoading trained YOLOv8 model...")

    model = YOLO(str(MODEL_PATH))

    image_path = get_test_image()

    print(f"Test image: {image_path.name}")

    print("\nRunning surgical instrument detection...")

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="prediction",
        exist_ok=True,
        verbose=False
    )

    # --------------------------------------------------
    # 6. DISPLAY DETECTION RESULTS
    # --------------------------------------------------

    result = results[0]

    print("\n" + "=" * 55)
    print("SURGICAL INSTRUMENT DETECTION RESULTS")
    print("=" * 55)

    if result.boxes is None or len(result.boxes) == 0:

        print("No surgical instruments detected.")

    else:

        for index, box in enumerate(result.boxes, start=1):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = CLASS_NAMES.get(
                class_id,
                "Unknown instrument"
            )

            coordinates = box.xyxy[0].tolist()

            print(f"\nDetection {index}")
            print(f"Instrument: {class_name}")
            print(f"Confidence: {confidence:.2%}")
            print(
                "Bounding box:",
                [round(value, 2) for value in coordinates]
            )

    print("\n" + "=" * 55)
    print("DETECTION COMPLETED")
    print("=" * 55)

    print(f"Original image: {image_path}")
    print(f"Output folder: {OUTPUT_DIR / 'prediction'}")


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():

    validate_files()
    detect_instruments()


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nDETECTION ERROR:")
        print(error)
        raise
