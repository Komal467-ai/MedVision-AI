"""ByteTrack-based persistent instrument tracking and annotated-video export."""

from pathlib import Path
from typing import Dict, List, Optional

import cv2


class SurgicalInstrumentTracker:
    """Use Ultralytics' ByteTrack integration to keep IDs across frames."""

    def __init__(self, weights: str = "models/best_yolo.pt",
                 confidence: float = 0.25, tracker_config: str = "bytetrack.yaml",
                 device: Optional[str] = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install Ultralytics: pip install ultralytics") from exc

        self.model = YOLO(weights)
        self.confidence = confidence
        self.tracker_config = tracker_config
        self.device = device

    def track_video(self, video_path: str, output_path: str = "results/tracking.mp4",
                    show: bool = False) -> List[Dict]:
        """Track objects and save an annotated MP4 plus per-frame records."""
        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        records: List[Dict] = []
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # persist=True is essential: ByteTrack retains track state
                # between consecutive frames in this video.
                result = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=self.confidence,
                    device=self.device,
                    verbose=False,
                )[0]

                annotated = result.plot()
                frame_objects = []
                if result.boxes is not None:
                    boxes = result.boxes
                    ids = boxes.id
                    for i, box in enumerate(boxes):
                        if ids is None:
                            track_id = None
                        else:
                            track_id = int(ids[i].item())
                        xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                        class_id = int(box.cls[0].item())
                        item = {
                            "frame": frame_index,
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": str(result.names[class_id]),
                            "confidence": float(box.conf[0].item()),
                            "bbox": xyxy,
                        }
                        frame_objects.append(item)
                        records.append(item)

                writer.write(annotated)
                if show:
                    cv2.imshow("Surgical instrument tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_index += 1
        finally:
            cap.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()

        return records
