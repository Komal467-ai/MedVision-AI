
"""
Prepare the Labeled Surgical Tools dataset for YOLOv8.

Uses the official training and testing lists.
Creates a validation split from the official training data.
"""

from pathlib import Path
import random
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "surgical_instruments" / "raw"
IMAGE_DIR = RAW_DIR / "Images" / "All" / "images"
LABEL_DIR = RAW_DIR / "Labels" / "label object names"
SPLIT_DIR = RAW_DIR / "Surgical-Dataset" / "Test-Train Groups"

OUTPUT_DIR = PROJECT_ROOT / "data" / "surgical_instruments" / "yolo"

TRAIN_LIST = SPLIT_DIR / "train-obj_detector.txt"
TEST_LIST = SPLIT_DIR / "test-obj_detector.txt"

CLASS_NAMES = [
    "scalpel_no4",
    "straight_dissection_clamp",
    "straight_mayo_scissor",
    "curved_mayo_scissor",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42


def read_split(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    names = []

    for line in file_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()

        if line:
            names.append(Path(line.replace("\\", "/")).name)

    return list(dict.fromkeys(names))


def build_image_index():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    image_index = {}

    for image in IMAGE_DIR.rglob("*"):
        if image.is_file() and image.suffix.lower() in VALID_EXTENSIONS:
            key = image.name.lower()

            if key in image_index:
                raise ValueError(
                    f"Duplicate image filename: {image.name}"
                )

            image_index[key] = image

    return image_index


def validate_annotation(label_path):
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {label_path}"
        )

    valid_lines = []

    lines = label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "expected 5 values"
            )

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(
                f"Invalid annotation in {label_path}, "
                f"line {line_number}"
            ) from error

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(
                f"Invalid class ID {class_id} in {label_path}"
            )

        x_center, y_center, width, height = values

        if not all(0 <= value <= 1 for value in values):
            raise ValueError(
                f"Coordinates must be between 0 and 1: {label_path}"
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid bounding-box dimensions: {label_path}"
            )

        if (
            x_center - width / 2 < -1e-6
            or x_center + width / 2 > 1.000001
        ):
            raise ValueError(
                f"Bounding box exceeds image width: {label_path}"
            )

        if (
            y_center - height / 2 < -1e-6
            or y_center + height / 2 > 1.000001
        ):
            raise ValueError(
                f"Bounding box exceeds image height: {label_path}"
            )

        valid_lines.append(line)

    return valid_lines


def prepare_split(split_name, image_names, image_index):
    image_output = OUTPUT_DIR / "images" / split_name
    label_output = OUTPUT_DIR / "labels" / split_name

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []
    class_counts = {
        i: 0 for i in range(len(CLASS_NAMES))
    }

    for image_name in image_names:
        source_image = image_index.get(image_name.lower())

        if source_image is None:
            missing.append(image_name)
            continue

        source_label = LABEL_DIR / f"{source_image.stem}.txt"

        valid_lines = validate_annotation(source_label)

        shutil.copy2(
            source_image,
            image_output / source_image.name
        )

        destination_label = (
            label_output / f"{source_image.stem}.txt"
        )

        destination_label.write_text(
            "\n".join(valid_lines) + ("\n" if valid_lines else ""),
            encoding="utf-8"
        )

        for line in valid_lines:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

        copied += 1

    print(f"\n{split_name.upper()} SPLIT")
    print(f"Images copied: {copied}")
    print(f"Missing images: {len(missing)}")

    if missing:
        print("First missing images:", missing[:10])

    print("Bounding boxes per class:")

    for class_id, count in class_counts.items():
        print(f"  {class_id}: {CLASS_NAMES[class_id]} = {count}")

    return copied


def write_yaml():
    dataset_path = OUTPUT_DIR.resolve().as_posix()

    yaml_content = f"""path: "{dataset_path}"
train: images/train
val: images/val
test: images/test

nc: 4

names:
  0: scalpel_no4
  1: straight_dissection_clamp
  2: straight_mayo_scissor
  3: curved_mayo_scissor
"""

    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\nCreated configuration: {yaml_path}")


def main():
    print("=" * 55)
    print("SURGICAL INSTRUMENT DATASET PREPARATION")
    print("=" * 55)

    required_paths = [
        IMAGE_DIR,
        LABEL_DIR,
        TRAIN_LIST,
        TEST_LIST,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path missing: {path}"
            )

    image_index = build_image_index()

    print(f"Images indexed: {len(image_index)}")

    train_names = read_split(TRAIN_LIST)
    test_names = read_split(TEST_LIST)

    train_set = {name.lower() for name in train_names}
    test_set = {name.lower() for name in test_names}

    overlap = train_set & test_set

    if overlap:
        raise ValueError(
            f"Official train/test split overlap: {len(overlap)} images"
        )

    random.seed(RANDOM_SEED)
    random.shuffle(train_names)

    validation_count = max(
        1,
        int(len(train_names) * VALIDATION_RATIO)
    )

    validation_names = train_names[:validation_count]
    actual_train_names = train_names[validation_count:]

    print(f"\nOfficial training list: {len(train_names)}")
    print(f"New training split: {len(actual_train_names)}")
    print(f"Validation split: {len(validation_names)}")
    print(f"Official testing list: {len(test_names)}")

    # This deletes only the generated YOLO dataset folder.
    # The original raw dataset is not deleted.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_count = prepare_split(
        "train",
        actual_train_names,
        image_index
    )

    val_count = prepare_split(
        "val",
        validation_names,
        image_index
    )

    test_count = prepare_split(
        "test",
        test_names,
        image_index
    )

    if not train_count or not val_count or not test_count:
        raise RuntimeError(
            "One or more dataset splits are empty."
        )

    write_yaml()

    print("\n" + "=" * 55)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 55)

    print(f"Training images:   {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Testing images:    {test_count}")
    print(f"Dataset location:  {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
