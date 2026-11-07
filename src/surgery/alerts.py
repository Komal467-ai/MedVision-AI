"""Rule-based surgical safety alerts (decision-support only).

These heuristics are demonstrators, not clinically validated bleeding or injury
detectors. A qualified human must review every alert.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class SurgicalSafetyMonitor:
    def __init__(self, red_ratio_threshold: float = 0.08,
                 proximity_pixels: float = 60.0,
                 phase_duration_limits: Optional[Dict[str, float]] = None):
        self.red_ratio_threshold = red_ratio_threshold
        self.proximity_pixels = proximity_pixels
        # Values are illustrative maximum durations in seconds. Replace them
        # with validated thresholds for the dataset and clinical workflow.
        self.phase_duration_limits = phase_duration_limits or {}
        self.phase_start_times = {}
        self.alert_history: List[Dict] = []

    def detect_bleeding(self, frame: np.ndarray, frame_index: int = 0) -> Optional[Dict]:
        """Flag unusually large red regions using HSV color thresholding."""
        if frame is None or frame.size == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red_1 = np.array([0, 70, 50])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 70, 50])
        upper_red_2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask |= cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_ratio = float(np.count_nonzero(mask)) / float(mask.size)

        if red_ratio >= self.red_ratio_threshold:
            return self._record({
                "frame": frame_index,
                "type": "possible_bleeding",
                "severity": "high" if red_ratio >= self.red_ratio_threshold * 1.5 else "medium",
                "value": red_ratio,
                "message": "Large red-colored region detected; human review required.",
            })
        return None

    @staticmethod
    def _center(box: Sequence[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def detect_proximity(self, detections: Iterable[Dict],
                         frame_index: int = 0) -> List[Dict]:
        """Flag instrument boxes close to boxes labelled as anatomy/tissue."""
        items = list(detections)
        instruments = [d for d in items if d.get("category") == "instrument"]
        anatomy = [d for d in items if d.get("category") == "anatomy"]
        alerts = []

        for instrument in instruments:
            for tissue in anatomy:
                ix, iy = self._center(instrument["bbox"])
                ax, ay = self._center(tissue["bbox"])
                distance = float(np.hypot(ix - ax, iy - ay))
                if distance <= self.proximity_pixels:
                    alerts.append(self._record({
                        "frame": frame_index,
                        "type": "instrument_anatomy_proximity",
                        "severity": "medium",
                        "distance_pixels": distance,
                        "message": "Instrument is close to an anatomy region; human review required.",
                    }))
        return alerts

    def update_phase(self, phase: str, timestamp_seconds: float,
                     frame_index: int = 0) -> Optional[Dict]:
        """Flag a phase that exceeds its configured maximum duration."""
        if phase not in self.phase_start_times:
            self.phase_start_times[phase] = timestamp_seconds
            return None

        duration = timestamp_seconds - self.phase_start_times[phase]
        maximum = self.phase_duration_limits.get(phase)
        if maximum is not None and duration > maximum:
            return self._record({
                "frame": frame_index,
                "type": "phase_duration_anomaly",
                "severity": "medium",
                "phase": phase,
                "duration_seconds": duration,
                "threshold_seconds": maximum,
                "message": f"Phase {phase} exceeded its configured duration threshold.",
            })
        return None

    def reset_phase_timer(self, phase: str, timestamp_seconds: float) -> None:
        """Call when a new phase segment starts, to measure each segment."""
        self.phase_start_times[phase] = timestamp_seconds

    def process_frame(self, frame: np.ndarray, detections: Iterable[Dict],
                      frame_index: int = 0) -> List[Dict]:
        """Run frame-level bleeding and proximity checks."""
        alerts = []
        bleeding = self.detect_bleeding(frame, frame_index)
        if bleeding:
            alerts.append(bleeding)
        alerts.extend(self.detect_proximity(detections, frame_index))
        return alerts

    def _record(self, alert: Dict) -> Dict:
        self.alert_history.append(alert)
        return alert
