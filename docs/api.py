"""FastAPI backend.   Run:  uvicorn api:app --host 0.0.0.0 --port 8000   (docs at /docs)"""
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import services

DISCLAIMER = "Research prototype for decision support only. Not a diagnostic tool."
IMG_EXT, VID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}, {".mp4", ".avi", ".mov"}
MAX_IMG, MAX_VID = 10 * 1024**2, 500 * 1024**2
JOB_DIR = Path("jobs")
JOB_DIR.mkdir(exist_ok=True)
JOBS: dict = {}  # in-memory job store (use Redis/DB for production)

app = FastAPI(title="AI Medical Analysis & Surgical Assistance API", version="1.0.0", description=DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def read_upload(file: UploadFile, exts: set, max_bytes: int) -> bytes:
    if Path(file.filename or "").suffix.lower() not in exts:
        raise HTTPException(415, f"Unsupported file type. Allowed: {sorted(exts)}")
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File too large (max {max_bytes // 1024**2} MB).")
    if not data:
        raise HTTPException(400, "Empty file.")
    return data


def run_service(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except services.ModelNotAvailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    out["disclaimer"] = DISCLAIMER
    return out


@app.get("/health")
def health():
    return {"status": "ok", "device": services.DEV}


@app.get("/models")
def models():
    """Which trained weights are present on the server."""
    return services.available()


@app.post("/predict/chest-xray")
def predict_cxr(file: UploadFile = File(...), gradcam: bool = True):
    return run_service(services.classify, "cxr", read_upload(file, IMG_EXT, MAX_IMG), gradcam)


@app.post("/predict/fracture")
def predict_fracture(file: UploadFile = File(...), gradcam: bool = True):
    return run_service(services.classify, "mura", read_upload(file, IMG_EXT, MAX_IMG), gradcam)


@app.post("/segment/brain-mri")
def segment_brain(file: UploadFile = File(...)):
    return run_service(services.segment_brain, read_upload(file, IMG_EXT, MAX_IMG))


# ---------- surgical video (async job) ----------
def _run_video_job(job_id: str, in_path: Path):
    job = JOBS[job_id]
    try:
        from video_pipeline import process  # imported lazily: heavy (ultralytics)
        out_path = JOB_DIR / f"{job_id}_out.mp4"
        df = process(str(in_path), str(services.weights_path("instruments")), str(out_path))
        csv_path = JOB_DIR / f"{job_id}_metrics.csv"
        df.to_csv(csv_path, index=False)
        ids = {i for lst in df["tracked_ids"] for i in lst}
        job.update(
            status="done", video=str(out_path), csv=str(csv_path),
            summary={
                "frames": int(len(df)),
                "duration_s": round(float(df["time_s"].max()), 2) if len(df) else 0,
                "unique_instruments_tracked": len(ids),
                "peak_blood_area_percent": round(float(df["blood_ratio"].max()) * 100, 2),
                "mean_blood_area_percent": round(float(df["blood_ratio"].mean()) * 100, 2),
                "note": "Blood area is a relative colour-based estimate, not a measured volume.",
            },
        )
    except Exception as e:  # report failure to the client instead of crashing the worker
        job.update(status="failed", error=str(e))
    finally:
        in_path.unlink(missing_ok=True)


@app.post("/video/jobs", status_code=202)
def create_video_job(background: BackgroundTasks, file: UploadFile = File(...)):
    data = read_upload(file, VID_EXT, MAX_VID)
    try:
        services.weights_path("instruments")
    except services.ModelNotAvailable as e:
        raise HTTPException(503, str(e))
    job_id = uuid.uuid4().hex
    in_path = JOB_DIR / f"{job_id}_in{Path(file.filename).suffix.lower()}"
    in_path.write_bytes(data)
    JOBS[job_id] = {"status": "processing"}
    background.add_task(_run_video_job, job_id, in_path)
    return {"job_id": job_id, "status": "processing", "poll": f"/video/jobs/{job_id}"}


def _get_job(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job id.")
    return JOBS[job_id]


@app.get("/video/jobs/{job_id}")
def video_job_status(job_id: str):
    job = _get_job(job_id)
    return {"job_id": job_id, "status": job["status"], "error": job.get("error"),
            "summary": job.get("summary"), "disclaimer": DISCLAIMER}


@app.get("/video/jobs/{job_id}/video")
def video_job_video(job_id: str):
    job = _get_job(job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"Job is {job['status']}.")
    return FileResponse(job["video"], media_type="video/mp4", filename="annotated.mp4")


@app.get("/video/jobs/{job_id}/metrics")
def video_job_metrics(job_id: str):
    job = _get_job(job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"Job is {job['status']}.")
    return FileResponse(job["csv"], media_type="text/csv", filename="video_metrics.csv")
