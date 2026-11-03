import streamlit as st
import requests
import json
from PIL import Image

# ---------- CONFIG ----------
API_URL = "http://localhost:8000"
USE_MOCK = True  # UI works standalone until M3/M4 models are ready

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stApp { background: #FFFFFF; }
    h1 { color: #0066FF !important; font-weight: 700; letter-spacing: -1px; }
    .metric-card {
        background: #F5F7FA; padding: 1.2rem; border-radius: 14px;
        border: 1px solid #E1E5EB; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .stButton>button {
        background: linear-gradient(90deg, #0066FF, #00A3FF);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.6rem; font-weight: 600; transition: 0.2s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,102,255,0.3); }
    .badge {
        display:inline-block; padding: 4px 12px; border-radius: 20px;
        background:#0066FF; color:#FFFFFF; font-weight:600; font-size:0.8rem;
    }
    .alert-box {
        background: #FFF3F3; border-left: 4px solid #E53935;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
        color: #B71C1C;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🩺 MediScan AI")
    st.caption("Multi-modal Medical AI Platform")
    st.divider()
    st.markdown("**Team**")
    st.markdown("M1 · M2 · M3 · M4 · **M5 (you)**")
    st.divider()
    st.markdown("**System Status**")
    if USE_MOCK:
        api_ok = True
    else:
        try:
            r = requests.get(f"{API_URL}/health", timeout=2)
            api_ok = r.status_code == 200
        except Exception:
            api_ok = False
    st.markdown(f"{'🟢' if api_ok else '🔴'} FastAPI backend")
    st.markdown(f"{'🟡' if USE_MOCK else '🟢'} Model mode: {'MOCK' if USE_MOCK else 'LIVE'}")
    st.divider()
    st.caption("v1.0 · Drop 3 skeleton")

# ---------- HEADER & TABS ----------
st.markdown("# 🩺 MediScan AI")
st.markdown("##### Unified diagnosis + surgery analysis dashboard")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔬 Diagnosis", "🎥 Surgery", "📄 Report"])

# ================= TAB 1: DIAGNOSIS =================
with tab1:
    st.markdown("### Chest X-Ray Analysis")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"], key="diagnosis_upload")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)
            run = st.button("Analyze X-Ray", key="run_diag")
        else:
            run = False

    with col2:
        st.markdown("#### Results")
        if uploaded and run:
            with st.spinner("Running CNN + U-Net + Grad-CAM..."):
                if USE_MOCK:
                    result = {
                        "classification": {"label": "Pneumonia", "confidence": 0.93},
                        "segmentation": {"dice": 0.84, "mask_url": None},
                        "gradcam_url": None,
                    }
                else:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    try:
                        r = requests.post(f"{API_URL}/predict/diagnosis", files=files, timeout=60)
                        result = r.json()
                    except Exception as e:
                        st.error(f"API error: {e}")
                        result = None

                if result:
                    cls = result["classification"]
                    st.markdown(
                        f'<div class="metric-card"><h4>Prediction</h4>'
                        f'<h2 style="color:#0066FF;margin:0;">{cls["label"]}</h2>'
                        f'<span class="badge">Confidence: {cls["confidence"]*100:.1f}%</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.metric("Segmentation Dice", f'{result["segmentation"]["dice"]:.3f}')
                    st.info("Grad-CAM overlay will appear here once M3 pushes the model.")
        else:
            st.caption("Upload an image and click **Analyze X-Ray**.")

# ================= TAB 2: SURGERY =================
with tab2:
    st.markdown("### Surgical Video Analysis")
    video_file = st.file_uploader("Upload a surgical video (mp4, avi, mov)", type=["mp4", "avi", "mov"], key="surgery_upload")
    if video_file:
        st.video(video_file)
        if st.button("Analyze Surgery", key="run_surg"):
            with st.spinner("Running YOLO + ByteTrack + Phase recognizer..."):
                if USE_MOCK:
                    result = {
                        "tracking_video_url": None,
                        "phases": [
                            {"name": "Preparation", "start": 0, "end": 120},
                            {"name": "Dissection", "start": 120, "end": 340},
                            {"name": "Suturing", "start": 340, "end": 600},
                        ],
                        "alerts": [{"type": "bleeding", "time": 145, "severity": "high", "msg": "Possible bleeding detected"}],
                    }
                else:
                    files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
                    try:
                        r = requests.post(f"{API_URL}/predict/surgery", files=files, timeout=300)
                        result = r.json()
                    except Exception as e:
                        st.error(f"API error: {e}")
                        result = None

                if result:
                    c1, c2 = st.columns([1, 1], gap="large")
                    with c1:
                        st.markdown("#### Phase Timeline")
                        for p in result["phases"]:
                            st.markdown(f"- **{p['name']}** — {p['start']}s to {p['end']}s")
                    with c2:
                        st.markdown("#### Alerts")
                        for a in result["alerts"]:
                            st.markdown(
                                f'<div class="alert-box">⚠️ <b>{a["type"].upper()}</b> at {a["time"]}s — {a["msg"]}</div>',
                                unsafe_allow_html=True,
                            )
    else:
        st.caption("Upload a surgical video to begin tracking + phase recognition.")

# ================= TAB 3: REPORT =================
with tab3:
    st.markdown("### Structured Report")
    st.caption("Generates a JSON report combining diagnosis + surgery results.")
    if st.button("Generate Sample Report"):
        sample = {
            "patient_id": "DEMO-001",
            "diagnosis": {"label": "Pneumonia", "confidence": 0.93},
            "surgery": {"phases": 3, "alerts": 1},
            "generated_at": "2025-01-01T12:00:00Z",
        }
        st.code(json.dumps(sample, indent=2), language="json")
        st.download_button("⬇️ Download JSON", data=json.dumps(sample, indent=2), file_name="report.json", mime="application/json")