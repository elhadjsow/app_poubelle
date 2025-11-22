import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import io
import gc
import os
import urllib.request
import tempfile
import cv2

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "model/poubelle_yolov8.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"  # Exemple d'URL

st.set_page_config(
    page_title="🗑️ SmartBin Detector", 
    layout="wide", 
    page_icon="🗑️",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CSS DESIGN
# -------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-container {
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    .upload-container:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .confidence-bar {
        height: 10px;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #1dd1a1);
        border-radius: 10px;
        margin: 10px 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .full-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .empty-badge {
        background: linear-gradient(135deg, #1dd1a1 0%, #10ac84 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem;
        width: 100%;
    }
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# FONCTIONS UTILITAIRES
# -------------------------------

def download_model():
    """Télécharge le modèle YOLO depuis l'URL"""
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
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.success(f"✅ Modèle disponible ({model_size:.1f} MB)")
        
        # Bouton pour télécharger le modèle localement
        with open(MODEL_PATH, "rb") as f:
            st.download_button(
                label="📥 Télécharger le modèle local",
                data=f,
                file_name="poubelle_yolov8.pt",
                mime="application/octet-stream",
                use_container_width=True
            )
        
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
    uploaded_file = st.file_uploader(
        "",
        type=['jpg','jpeg','png'],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# TRAITEMENT IMAGE
# -------------------------------
if uploaded_file is not None and check_model_exists():
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGB")
    MAX_SIZE = (1024, 1024)
    img.thumbnail(MAX_SIZE)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(img, use_container_width=True, caption="Image originale")

    # Prédiction
    with st.spinner("🔍 Analyse en cours..."):
        box, pred, score = predict_image_yolo(img_array)

    # Affichage des résultats
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if pred == "aucune détection":
        st.error("🚫 Aucune poubelle détectée")
    elif pred == "erreur":
        st.error("❌ Erreur de détection")
    else:
        badge_color = "full-badge" if pred=="pleine" else "empty-badge"
        st.markdown(f'<div class="{badge_color}">🗑️ Poubelle {pred.upper()}</div>', unsafe_allow_html=True)
        st.metric("Score de confiance", f"{score:.2%}")

        # Dessiner box
        draw = ImageDraw.Draw(img)
        x, y, w, h = box
        color = "#ff6b6b" if pred=="pleine" else "#1dd1a1"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=6)
        caption = f"Poubelle {pred} (confiance: {score:.2%})"
        st.image(img, use_container_width=True, caption=caption)
    st.markdown('</div>', unsafe_allow_html=True)

elif uploaded_file is not None and not check_model_exists():
    st.error("❌ Impossible de traiter l'image : le modèle n'est pas disponible")
    st.info("📥 Veuillez télécharger le modèle depuis la sidebar pour continuer")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "🚀 Propulsé par YOLOv8 | SmartBin Detector v1.0"
    "</div>", 
    unsafe_allow_html=True
)
