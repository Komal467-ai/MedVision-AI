import streamlit as st
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
import sys
import os

# Add path so we can import src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.image_preprocess import preprocess_image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MedVision-AI ✨", 
    page_icon="🩺", 
    layout="wide"
)

# --- CUSTOM CSS (The Gen-Z Aesthetic Magic) ---
st.markdown("""
<style>
    /* Import a cute modern font */
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
    }

    /* Soft pink/purple gradient background for the whole app */
    .stApp {
        background: linear-gradient(135deg, #FFF0F5 0%, #F3E8FF 100%);
    }

    /* Make the sidebar look like a frosted glass card */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-right: 2px solid #FFD1DC;
    }

    /* Rounded corners for images */
    [data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(255, 182, 193, 0.3);
        border: 2px solid white;
    }

    /* Style the file uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF;
        border: 2px dashed #D8B4E2;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #FFB6C1;
        background-color: #FFF9FA;
    }

    /* Custom Title Gradient Text */
    .title-text {
        font-size: 3.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF69B4, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle-text {
        text-align: center;
        color: #8B7A9E;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }

    /* Success message aesthetic */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<p class="title-text">MedVision-AI 🩺✨</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Smart Medical Image Analysis • Preprocessing & Inference Pipeline</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🌸 About This Prototype")
    st.info(
        "**Working Pipeline Demo**\n\n"
        "✅ Preprocessing: Denoise + CLAHE\n\n"
        "✅ Inference: EfficientNet-B0\n\n"
        "*(Status: Model is not yet fine-tuned on medical data. Fine-tuning starts next week!)*"
    )
    
    st.markdown("### 🎀 Pipeline Steps")
    st.markdown(
        "1️⃣ Grayscale conversion\n\n"
        "2️⃣ Non-Local Means denoising\n\n"
        "3️⃣ CLAHE contrast enhancement\n\n"
        "4️⃣ Resize to 224×224\n\n"
        "5️⃣ Normalize to [0, 1]\n\n"
        "6️⃣ Stack to 3-channel for CNN"
    )
    st.markdown("---")
    st.caption("Made with 💖 by Team MedVision")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights="DEFAULT")
    model.eval()
    return model

model = load_model()

IMAGENET_LABELS = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich",
]

# --- TABS ---
tab1, tab2 = st.tabs(["🔬 Upload & Analyze", "📖 Project Info"])

with tab1:
    st.markdown("### 📸 Upload a Medical Image")
    st.markdown("Drop an X-ray, CT, or MRI scan below to see the magic happen! ✨")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="X-ray, CT scan, or MRI image (JPEG/PNG)",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📷 Original Image")
            st.image(image, use_container_width=True)

        with st.spinner("Preprocessing (Denoise + CLAHE)... 🪄"):
            preprocessed, enhanced = preprocess_image(image_np)

        with col2:
            st.markdown("#### ✨ Preprocessed (Denoise + CLAHE)")
            st.image(enhanced, use_container_width=True, clamp=True)

        st.markdown("---")
        st.markdown("### 🧠 Model Inference")
        
        with st.spinner("Analyzing image... 🔍"):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_idx = torch.topk(probabilities, 3)

        st.success("🎉 Inference complete — pipeline is working end-to-end!")

        st.markdown("**Top-3 predictions (ImageNet pretrained model):**")
        
        # Styled predictions
        for i, (prob, idx) in enumerate(zip(top_prob, top_idx)):
            label = (
                IMAGENET_LABELS[idx.item()]
                if idx.item() < len(IMAGENET_LABELS)
                else f"class_{idx.item()}"
            )
            st.write(f"**{i + 1}.** `{label}` — **{prob.item() * 100:.2f}%**")

        st.warning(
            "💡 **Note:** The model is currently ImageNet-pretrained (not medical). "
            "Next week: we fine-tune on the Chest X-Ray Pneumonia dataset "
            "for real NORMAL/PNEUMONIA predictions!"
        )

        st.markdown("---")
        st.markdown("#### 🔧 What the pipeline does")
        st.table({
            "Step": ["1", "2", "3", "4", "5", "6"],
            "Operation": ["Grayscale conversion", "Non-Local Means denoising", "CLAHE contrast enhancement", "Resize 224×224", "Normalize [0,1]", "Stack to 3-channel"],
            "Purpose": ["Standard for X-ray/CT", "Removes noise, preserves edges", "Local contrast enhancement", "EfficientNet input", "Neural network input", "Pretrained backbone"]
        })

with tab2:
    st.markdown("### 🌟 About MedVision-AI")
    st.markdown("""
    We are building an AI-based healthcare computer vision system with two core modules:
    
    **🩺 Module 1 — Diagnosis (Current Prototype)**
    - CNN classification for X-ray/CT/MRI abnormalities
    - U-Net segmentation for lesion localization
    - Grad-CAM explainability
    
    **🔪 Module 2 — Surgical Assistance (Coming next!)**
    - YOLOv8 instrument detection
    - ByteTrack instrument tracking
    - LSTM surgical phase recognition
    - Safety alerts
    
    **🚀 Current Status**
    - ✅ Preprocessing pipeline working
    - ✅ Inference pipeline working
    - ⏳ CNN fine-tuning on Chest X-Ray dataset (next week)
    - ⏳ U-Net segmentation (Week 4)
    - ⏳ Surgical module (Week 5–6)
    
    **👩‍💻 The Team**
    - **Komal** — Team Lead + Integration
    - **Radhika** — Data Engineer
    - **Megha** — ML Engineer (Diagnosis)
    - **[Member 4]** — ML Engineer (Surgery)
    - **[Member 5]** — UI + Deployment
    
    **🛠️ Tech Stack**
    `PyTorch` • `torchvision` • `OpenCV` • `Streamlit` • `FastAPI` • `Docker`
    """)
    st.error("⚠️ **Disclaimer:** This system is decision-support only, not autonomous diagnosis. All outputs require clinician review.")

st.markdown("---")
st.caption("MedVision-AI — Drop 1 Prototype — Made with 💖")