# M4 Surgical Assistance module

## Scope
M4 supplies four model components: YOLOv8 detection, ByteTrack tracking,
CNN+LSTM phase recognition, and rule-based safety-alert heuristics. M4 does
not own FastAPI, Streamlit, Docker, or final API orchestration.

## Integration contract for M5
- `SurgicalInstrumentDetector.predict_frame(frame)` returns a list of
  `{class_id, class_name, confidence, bbox}` records.
- `SurgicalInstrumentTracker.track_video(video_path, output_path)` returns
  per-frame records containing `frame`, `track_id`, `class_name`, `confidence`,
  and `bbox`.
- `PhaseRecognizer.predict(clip)` accepts `[T, 3, H, W]` and returns
  `{phase, confidence}`.
- `SurgicalSafetyMonitor.process_frame(frame, detections, frame_index)` returns
  zero or more alert dictionaries.

For proximity alerts, enrich detections with `category: "instrument"` or
`category: "anatomy"` before calling the monitor. The current YOLO class list
only contains instruments, so anatomy localization needs an anatomy detector
or segmentation source from the wider project.

## Important limitations
- `best_yolo.pt` and `best_phase.pth` are trained artifacts; they are not
  included in this source bundle. Generate them using the approved datasets.
- The seven phase names and instrument names must match the labels in your
  actual dataset.
- Red-color thresholding is only a visual heuristic; it is not a validated
  clinical bleeding detector.
- Phase duration thresholds must be configured from evidence, not guessed.
- MOTA requires ground-truth identity annotations and a tracking evaluation
  tool. A successful MP4 alone does not establish MOTA >= 0.60.
- Detection mAP must be computed on a held-out labelled test set.

## Training
1. Put the YOLO dataset in the structure documented in `docs/yolo_format.md`.
2. Update `surgical_instruments.yaml` to match actual classes and paths.
3. Run `SurgicalInstrumentDetector.train(...)` on a GPU.
4. Copy the resulting `best.pt` to `models/best_yolo.pt`.
5. Build phase clips from labelled Cholec80 phase intervals; train the
   `PhaseCNNLSTM` with `train_phase_model(...)`.
6. Save the best phase checkpoint to `models/best_phase.pth`.

## Run
```python
from src.surgery.detector import SurgicalInstrumentDetector
from src.surgery.tracker import SurgicalInstrumentTracker

detector = SurgicalInstrumentDetector("models/best_yolo.pt")
tracker = SurgicalInstrumentTracker("models/best_yolo.pt")
records = tracker.track_video("sample_surgery.mp4", "results/tracking.mp4")
print(records[:3])
```
