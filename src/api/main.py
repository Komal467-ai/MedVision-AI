from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil

# Import M4's surgical functions
try:
    from src.surgery.detector import SurgicalInstrumentDetector
    from src.surgery.tracker import SurgicalInstrumentTracker
    from src.surgery.alerts import SurgicalSafetyMonitor
    M4_AVAILABLE = True
except ImportError:
    M4_AVAILABLE = False
    print("Warning: M4 surgery modules not found. Using mock data.")

app = FastAPI(title="MediScan AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI is running"}

@app.post("/predict/diagnosis")
async def predict_diagnosis(file: UploadFile = File(...)):
    # TODO: M3 will replace this with real model calls
    return {
        "classification": {"label": "Pneumonia (Dummy)", "confidence": 0.93},
        "segmentation": {"dice": 0.84, "mask_url": None},
        "gradcam_url": None
    }

@app.post("/predict/surgery")
async def predict_surgery(file: UploadFile = File(...)):
    # Save the uploaded video temporarily
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check if M4 model weights actually exist
    model_path = "models/best_yolo.pt"
    
    if M4_AVAILABLE and os.path.exists(model_path):
        try:
            # REAL M4 INTEGRATION (Jab M2 model de dega tab yeh chalega)
            tracker = SurgicalInstrumentTracker(model_path)
            output_path = "results/tracking_temp.mp4"
            
            # Run tracking
            records = tracker.track_video(temp_video_path, output_path)
            
            os.remove(temp_video_path) # Clean up
            
            return {
                "status": "success",
                "message": "Real M4 model executed.",
                "tracking_video_url": output_path,
                "phases": [{"name": "Real Phase 1", "start": 0, "end": 120}],
                "alerts": []
            }
        except Exception as e:
            os.remove(temp_video_path)
            return {"error": f"Model failed to run: {str(e)}"}
    
    else:
        # FALLBACK TO MOCK (Kyunki abhi models/best_yolo.pt nahi hai)
        os.remove(temp_video_path)
        return {
            "status": "mock",
            "message": "M4 model weights not found yet. Returning mock data.",
            "tracking_video_url": None,
            "phases": [
                {"name": "Preparation", "start": 0, "end": 120},
                {"name": "Dissection", "start": 120, "end": 340},
                {"name": "Suturing", "start": 340, "end": 600}
            ],
            "alerts": [{"type": "bleeding", "time": 145, "severity": "high", "msg": "Possible bleeding detected"}]
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)