import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import gc
import os
import urllib.request
import cv2

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "model/poubelle_yolov8.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"

st.set_page_config(
    page_title="🗑️ SmartBin Detector", 
    layout="wide", 
    page_icon="🗑️",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CSS DESIGN MODERNE BLEU
# -------------------------------
st.markdown("""
<style>
/* HEADER */
.main-header {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #1E90FF 0%, #00BFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.3rem;
    color: #4682B4;
    text-align: center;
    margin-bottom: 2rem;
}

/* UPLOAD */
.upload-container {
    border: 2px dashed #1E90FF;
    border-radius: 25px;
    padding: 3rem;
    text-align: center;
    background: rgba(30, 144, 255, 0.05);
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(30,144,255,0.1);
    transition: all 0.3s ease;
}
.upload-container:hover {
    background: rgba(30, 144, 255, 0.15);
    border-color: #00BFFF;
}

/* RESULT CARD */
.result-card {
    background: linear-gradient(145deg, #f0f8ff 0%, #e6f0ff 100%);
    border-radius: 25px;
    padding: 2rem;
    margin: 1rem 0;
    border-left: 8px solid #1E90FF;
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.result-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.25);
}

/* CONFIDENCE BAR */
.confidence-bar {
    height: 12px;
    background: linear-gradient(90deg, #1E90FF, #00BFFF, #4682B4);
    border-radius: 10px;
    margin: 10px 0;
}

/* STAT CARD */
.stat-card {
    background: linear-gradient(135deg, #1E90FF 0%, #4682B4 100%);
    color: white;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    margin: 0.5rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

/* BADGES */
.full-badge {
    background: linear-gradient(135deg, #FF4500 0%, #FF6347 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 5px 15px rgba(255,69,0,0.3);
}
.empty-badge {
    background: linear-gradient(135deg, #00CED1 0%, #20B2AA 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,206,209,0.3);
}

/* BOUTONS */
.download-btn {
    background: linear-gradient(135deg, #1E90FF 0%, #4682B4 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 12px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 0.5rem 0;
    width: 100%;
}
.download-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(30,144,255,0.5);
}

/* FOOTER */
.footer {
    text-align:center;
    color:#4682B4;
    margin-top:2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# FONCTIONS
# -------------------------------
def download_model():
    try:
        os.makedirs("model", exist_ok=True)
        progress_text = st.sidebar.info("📥 Téléchargement du modèle en cours...")
        progress_bar = st.sidebar.progress(0)
        def update_progress(block_num, block_size, total_size):
            if total_size > 0:
                progress = min(block_num * block_size / total_size, 1.0)
                progress_bar.progress(progress)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=update_progress)
        progress_bar.progress(1.0)
        progress_text.empty()
        st.sidebar.success("✅ Modèle téléchargé avec succès !")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Erreur lors du téléchargement: {e}")
        return False

def check_model_exists():
    return os.path.exists(MODEL_PATH)

def predict_image_yolo(img_array):
    try:
        model = YOLO(MODEL_PATH)
        results = model(img_array)
        boxes = results[0].boxes
        if len(boxes) == 0:
            del model; gc.collect()
            return None, "aucune détection", 0.0
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        label = "pleine" if label_id == 0 else "vide"
        box_tuple = (x1, y1, x2 - x1, y2 - y1)
        del model; gc.collect()
        return box_tuple, label, score
    except Exception as e:
        st.error(f"Erreur YOLO : {e}")
        return None, "erreur", 0.0

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<h1 class="main-header">🗑️ SmartBin Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Détection intelligente des poubelles par Intelligence Artificielle</p>', unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.markdown("### 🛠️ Gestion du modèle")
    if check_model_exists():
        model_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        st.success(f"✅ Modèle disponible ({model_size:.1f} MB)")
        with open(MODEL_PATH, "rb") as f:
            st.download_button("📥 Télécharger le modèle local", f, "poubelle_yolov8.pt", use_container_width=True)
        if st.button("🔄 Retélécharger le modèle", use_container_width=True):
            if download_model():
                st.rerun()
    else:
        st.error("❌ Modèle non trouvé")
        if st.button("📥 Télécharger le modèle YOLO", use_container_width=True):
            if download_model():
                st.rerun()

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown("### 📤 Importez votre image")
st.markdown("Glissez-déposez ou sélectionnez une image contenant une poubelle")
if not check_model_exists():
    st.warning("⏳ Veuillez d'abord télécharger le modèle pour activer la détection")
    uploaded_file = None
else:
    uploaded_file = st.file_uploader("", type=['jpg','jpeg','png'], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# TRAITEMENT IMAGE
# -------------------------------
if uploaded_file and check_model_exists():
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((1024,1024))
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    st.image(img, use_container_width=True, caption="Image originale")
    with st.spinner("🔍 Analyse en cours..."):
        box, pred, score = predict_image_yolo(img_array)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if pred=="aucune détection":
        st.error("🚫 Aucune poubelle détectée")
    elif pred=="erreur":
        st.error("❌ Erreur de détection")
    else:
        badge_color = "full-badge" if pred=="pleine" else "empty-badge"
        st.markdown(f'<div class="{badge_color}">🗑️ Poubelle {pred.upper()}</div>', unsafe_allow_html=True)
        st.metric("Score de confiance", f"{score:.2%}")
        draw = ImageDraw.Draw(img)
        x,y,w,h = box
        color = "#FF4500" if pred=="pleine" else "#00CED1"
        draw.rectangle([x,y,x+w,y+h], outline=color, width=6)
        st.image(img, use_container_width=True, caption=f"Poubelle {pred} (confiance: {score:.2%})")
    st.markdown('</div>', unsafe_allow_html=True)
elif uploaded_file and not check_model_exists():
    st.error("❌ Impossible de traiter l'image : le modèle n'est pas disponible")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("<div class='footer'>🚀 Propulsé par YOLOv8 | SmartBin Detector v1.0</div>", unsafe_allow_html=True)
