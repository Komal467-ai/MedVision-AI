"""YOLOv8 training and inference for surgical-instrument detection.

Expected dataset format: standard Ultralytics YOLO detection format.
The YAML file must define train/val paths and class names.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class SurgicalInstrumentDetector:
    """Thin, reusable wrapper around an Ultralytics YOLO detector."""

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.25,
                 image_size: int = 640, device: Optional[str] = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Install Ultralytics first: pip install ultralytics"
            ) from exc

        self.model = YOLO(weights)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    @staticmethod
    def train(data_yaml: str, base_weights: str = "yolov8n.pt",
              epochs: int = 60, image_size: int = 640,
              batch_size: int = 16, project: str = "runs/surgery",
              run_name: str = "instrument_detection",
              device: Optional[str] = None) -> str:
        """Fine-tune YOLOv8 and return the path to best.pt.

        Training data must be labelled; this function does not create labels.
        """
        from ultralytics import YOLO

        if not Path(data_yaml).is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

        model = YOLO(base_weights)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=project,
            name=run_name,
            device=device,
            patience=15,
            workers=4,
            plots=True,
            save=True,
        )
        best_path = Path(results.save_dir) / "weights" / "best.pt"
        if not best_path.exists():
            raise RuntimeError(f"Training finished but best weights are missing: {best_path}")
        return str(best_path)

    def predict_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Return detections in a stable JSON-friendly format."""
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a valid OpenCV/numpy image")

        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        names = result.names
        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            class_id = int(box.cls[0].item())
            detections.append({
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "confidence": float(box.conf[0].item()),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        return detections

    def evaluate(self, data_yaml: str) -> Dict[str, float]:
        """Run Ultralytics validation and return common detection metrics."""
        metrics = self.model.val(data=data_yaml, imgsz=self.image_size,
                                 device=self.device, verbose=False)
        box_metrics = metrics.box
        return {
            "map50": float(box_metrics.map50),
            "map50_95": float(box_metrics.map),
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
        }

    def annotate_frame(self, frame: np.ndarray,
                       detections: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """Draw predicted bounding boxes and labels on a copy of the frame."""
        output = frame.copy()
        detections = detections if detections is not None else self.predict_frame(frame)
        for item in detections:
            x1, y1, x2, y2 = map(int, item["bbox"])
            label = f'{item["class_name"]} {item["confidence"]:.2f}'
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(output, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        return output
