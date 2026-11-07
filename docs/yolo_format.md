# Surgical instrument dataset format

M4 uses Ultralytics YOLO detection format. Each image must have a matching
`.txt` label file with one object per line:

```text
class_id x_center y_center width height
```

All coordinates are normalized to the range 0–1 relative to image width/height.
Class IDs are zero-based and must match `surgical_instruments.yaml`.

Recommended directory layout:

```text
data/surgical_instruments/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Do not commit Cholec80 or other restricted datasets, patient information, or
unlicensed frames to GitHub. Keep data outside the repository and document
approved access instructions.

Example training:

```python
from src.surgery.detector import SurgicalInstrumentDetector

best_weights = SurgicalInstrumentDetector.train(
    data_yaml="surgical_instruments.yaml",
    epochs=60,
    image_size=640,
)
print("Best checkpoint:", best_weights)
```

The class names above are placeholders. Confirm them against the actual labels
before training; incorrect class mappings invalidate evaluation.
