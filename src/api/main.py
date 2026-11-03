from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="MediScan AI API", version="1.0")

# Allow Streamlit to talk to FastAPI
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
    # TODO: M3 will replace this with real model calls in Week 5
    return {
        "classification": {"label": "Pneumonia (Dummy)", "confidence": 0.93},
        "segmentation": {"dice": 0.84, "mask_url": None},
        "gradcam_url": None
    }

@app.post("/predict/surgery")
async def predict_surgery(file: UploadFile = File(...)):
    # TODO: M4 will replace this with real model calls in Week 6
    return {
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