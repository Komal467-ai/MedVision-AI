from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil

# Import M3 and M4 modules
try:
    from src.diagnosis.inference import predict_classification
    from src.surgery.tracker import SurgicalInstrumentTracker
    M3_AVAILABLE = True
    M4_AVAILABLE = True
except ImportError:
    M3_AVAILABLE = False
    M4_AVAILABLE = False

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
    if not M3_AVAILABLE:
        raise HTTPException(status_code=500, detail="M3 Diagnosis code not found.")
    
    try:
        contents = await file.read()
        # Real M3 inference call
        result = predict_classification(contents)
        return result
    except Exception as e:
        # Ab koi dummy nahi, seedha error aayega
        raise HTTPException(status_code=503, detail=f"M3 Model Error: {str(e)}")

@app.post("/predict/surgery")
async def predict_surgery(file: UploadFile = File(...)):
    if not M4_AVAILABLE:
        raise HTTPException(status_code=500, detail="M4 Surgery code not found.")

    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model_path = "models/best_yolo.pt"
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError("best_yolo.pt not found in models folder.")
        
        tracker = SurgicalInstrumentTracker(model_path)
        output_path = "results/tracking_temp.mp4"
        records = tracker.track_video(temp_video_path, output_path)
        os.remove(temp_video_path)
        
        return {
            "status": "success",
            "tracking_video_url": output_path,
            "phases": [{"name": "Real Phase 1", "start": 0, "end": 120}],
            "alerts": []
        }
    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise HTTPException(status_code=503, detail=f"M4 Model Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)