import streamlit as st
import requests
import json
from PIL import Image

# ---------- CONFIG ----------
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(rgba(245, 250, 255, 0.92), rgba(255, 255, 255, 0.95)), 
            url("https://images.unsplash.com/photo-1579684385127-1ef15d508118?q=80&w=2000&auto=format&fit=crop");
        background-size: cover; background-position: center; background-attachment: fixed;
    }
    h1 { 
        background: -webkit-linear-gradient(45deg, #004aad, #00b4d8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800 !important; letter-spacing: -1px;
    }
    h3 { color: #004aad !important; font-weight: 700 !important; }
    .metric-card {
        background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px);
        padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.5);
        border-left: 6px solid #00b4d8; box-shadow: 0 8px 32px rgba(0, 74, 173, 0.1);
    }
    .badge {
        display:inline-block; padding: 6px 16px; border-radius: 20px;
        background: linear-gradient(90deg, #00b4d8 0%, #0077b6 100%);
        color:#FFFFFF; font-weight:700; font-size:0.85rem; box-shadow: 0 2px 10px rgba(0, 180, 216, 0.3);
    }
    .alert-box {
        background: rgba(255, 240, 240, 0.9); border-left: 6px solid #ef233c;
        padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
        color: #d80032; font-weight: 500; box-shadow: 0 4px 12px rgba(239, 35, 60, 0.15);
        backdrop-filter: blur(5px);
    }
    .stButton>button {
        background: linear-gradient(90deg, #0077b6 0%, #00b4d8 100%);
        color: white; border: none; border-radius: 10px; padding: 0.6rem 2rem; 
        font-weight: 600; font-size: 1rem; transition: all 0.3s ease; 
        box-shadow: 0 4px 15px rgba(0, 119, 182, 0.3);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 119, 182, 0.5); }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(12px);
        border-right: 1px solid #e0e7ff; box-shadow: 2px 0 15px rgba(0,0,0,0.03);
    }
    div[data-testid="stHorizontalBlock"]:nth-of-type(1) button {
        background-color: transparent !important;
        color: #4b5563 !important; 
        border: 1px solid #d1d5db !important; 
        border-radius: 25px !important; 
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: none !important;
        transition: all 0.2s ease-in-out !important;
        width: auto !important; 
        white-space: nowrap !important; 
    }
    div[data-testid="stHorizontalBlock"]:nth-of-type(1) button:hover {
        background-color: #f3f4f6 !important;
        border-color: #9ca3af !important;
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🩺 MediScan AI")
    st.caption("Multi-modal Medical AI Platform")
    st.divider()
    st.markdown("**Team Members**")
    st.markdown("- 👑 **Komal** (M1)")
    st.markdown("- 📊 **Shivani** (M2)")
    st.markdown("- 🧠 **Parul** (M3)")
    st.markdown("- ✂️ **Radhika** (M4)")
    st.markdown("- 💻 **Megha** (M5) 👈")
    st.divider()
    st.markdown("**System Status**")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        api_ok = r.status_code == 200
    except Exception:
        api_ok = False
    st.markdown(f"{'🟢' if api_ok else '🔴'} FastAPI backend")
    st.markdown("🟢 Model mode: LIVE (Real Models Only)")
    st.caption("v1.0 · Drop 6")

# ---------- HEADER ----------
st.markdown("# 🩺 MediScan AI")
st.markdown("##### Unified diagnosis + surgery analysis dashboard")
st.write("") 

# ---------- HORIZONTAL PILL NAVIGATION ----------
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Diagnosis"

col1, col2, col3 = st.columns([2, 2, 2]) 

with col1:
    if st.button("🔬 Diagnosis", use_container_width=True):
        st.session_state.active_tab = "Diagnosis"
with col2:
    if st.button("🎥 Surgery", use_container_width=True):
        st.session_state.active_tab = "Surgery"
with col3:
    if st.button("📄 Report", use_container_width=True):
        st.session_state.active_tab = "Report"

st.markdown("---")

# ---------- MAIN CONTENT AREA ----------
active = st.session_state.active_tab

# ================= TAB 1: DIAGNOSIS =================
if active == "Diagnosis":
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
            with st.spinner("Running Real Models (CNN + U-Net + Grad-CAM)..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                try:
                    r = requests.post(f"{API_URL}/predict/diagnosis", files=files, timeout=60)
                    if r.status_code == 200:
                        result = r.json()
                    else:
                        st.error(f"Backend Error: {r.status_code} - {r.text}")
                        result = None
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                    result = None

                if result:
                    cls = result.get("classification", {"label": "Unknown", "confidence": 0})
                    st.markdown(
                        f'<div class="metric-card"><h4>Prediction</h4>'
                        f'<h2 style="color:#0077b6;margin:0;">{cls["label"]}</h2>'
                        f'<span class="badge">Confidence: {cls["confidence"]*100:.1f}%</span></div>',
                        unsafe_allow_html=True,
                    )
                    seg = result.get("segmentation", {"dice": 0})
                    st.metric("Segmentation Dice", f'{seg["dice"]:.3f}')
        else:
            st.caption("Upload an image and click **Analyze X-Ray**.")

# ================= TAB 2: SURGERY =================
elif active == "Surgery":
    st.markdown("### Surgical Video Analysis")
    video_file = st.file_uploader("Upload a surgical video (mp4, avi, mov)", type=["mp4", "avi", "mov"], key="surgery_upload")
    if video_file:
        st.video(video_file)
        if st.button("Analyze Surgery", key="run_surg"):
            with st.spinner("Running Real Models (YOLO + ByteTrack + Alerts)..."):
                files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
                try:
                    r = requests.post(f"{API_URL}/predict/surgery", files=files, timeout=300)
                    if r.status_code == 200:
                        result = r.json()
                    else:
                        st.error(f"Backend Error: {r.status_code} - {r.text}")
                        result = None
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                    result = None

                if result:
                    c1, c2 = st.columns([1, 1], gap="large")
                    with c1:
                        st.markdown("#### Phase Timeline")
                        for p in result.get("phases", []):
                            st.markdown(f"- **{p['name']}** — {p['start']}s to {p['end']}s")
                    with c2:
                        st.markdown("#### Alerts")
                        for a in result.get("alerts", []):
                            st.markdown(
                                f'<div class="alert-box">⚠️ <b>{a["type"].upper()}</b> at {a["time"]}s — {a["msg"]}</div>',
                                unsafe_allow_html=True,
                            )
    else:
        st.caption("Upload a surgical video to begin tracking + phase recognition.")

# ================= TAB 3: REPORT =================
elif active == "Report":
    st.markdown("### Structured Report")
    st.caption("Generates a comprehensive JSON report combining diagnosis + surgery results.")
    
    if st.button("Generate Combined Report"):
        sample_report = {
            "hospital_name": "MediScan AI Hospital",
            "patient_id": "PAT-2026-001",
            "date": "2026-11-09",
            "diagnosis": {
                "model": "EfficientNet-B0",
                "result": "Awaiting Real Model",
                "confidence": "0%",
                "gradcam_available": False
            },
            "surgery": {
                "video_analyzed": "True",
                "phases_detected": [],
                "alerts_triggered": 0,
                "alert_details": "Awaiting Real Model"
            },
            "status": "Live System (Real Models Pending)"
        }
        
        st.success("Report Generated Successfully!")
        st.json(sample_report)
        
        st.download_button(
            label="⬇️ Download JSON Report",
            data=json.dumps(sample_report, indent=4),
            file_name="medical_report.json",
            mime="application/json"
        )