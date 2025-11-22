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
# CSS DESIGN PREMIUM
# -------------------------------
st.markdown(""" 
<style>
/* ... ton CSS complet ici ... */
</style>
""", unsafe_allow_html=True)

# -------------------------------
# FONCTIONS UTILES
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

@st.cache_resource
def load_model(path):
    """Charge le modèle YOLO une seule fois en mémoire."""
    return YOLO(path)

def predict_image_yolo(img_array, model):
    """Prédiction d'une image avec le modèle déjà chargé."""
    try:
        results = model(img_array)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None, "aucune détection", 0.0
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        label = "pleine" if label_id == 0 else "vide"
        box_tuple = (x1, y1, x2 - x1, y2 - y1)
        return box_tuple, label, score
    except Exception as e:
        st.error(f"Erreur YOLO : {e}")
        return None, "erreur", 0.0

# -------------------------------
# HEADER PRINCIPAL
# -------------------------------
st.markdown("""
<div class="main-container">
    <h1 class="main-header">🗑️ SmartBin Detector</h1>
    <p class="sub-header">Détection intelligente des poubelles par Intelligence Artificielle</p>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR GESTION DU MODELE
# -------------------------------
with st.sidebar:
    st.markdown("<h3 style='color: white;'>🛠️ Gestion du Modèle</h3>", unsafe_allow_html=True)

    if check_model_exists():
        model_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        st.success(f"✅ Modèle disponible  \n*Taille: {model_size:.1f} MB*")
        with open(MODEL_PATH, "rb") as f:
            st.download_button("📥 Télécharger le modèle local", f, "poubelle_yolov8.pt", use_container_width=True)

        if st.button("🔄 Retélécharger le modèle", use_container_width=True):
            if download_model():
                st.rerun()

        model = load_model(MODEL_PATH)  # ⚡ modèle chargé une seule fois
    else:
        st.error("❌ Modèle non trouvé")
        if st.button("📥 Télécharger le modèle YOLO", use_container_width=True):
            if download_model():
                st.rerun()

# -------------------------------
# UPLOAD ZONE
# -------------------------------
st.markdown("""
<div class="upload-container">
    <h3 style='color: #667eea;'>📤 Importez votre image</h3>
    <p style='color: #6c757d;'>Glissez-déposez ou sélectionnez une image contenant une poubelle</p>
</div>
""", unsafe_allow_html=True)

if not check_model_exists():
    st.warning("⏳ Veuillez d'abord télécharger le modèle dans la sidebar pour activer la détection")
    uploaded_file = None
else:
    uploaded_file = st.file_uploader("", type=['jpg','jpeg','png'], label_visibility="collapsed")

# -------------------------------
# TRAITEMENT ET AFFICHAGE
# -------------------------------
if uploaded_file and check_model_exists():
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🖼️ Image Originale")
        st.image(img, use_container_width=True, caption="Votre image importée")

    with st.spinner("🔍 Analyse en cours..."):
        box, pred, score = predict_image_yolo(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), model)

    with col2:
        st.markdown("### 📊 Résultats")
        if pred == "aucune détection":
            st.error("🚫 Aucune Poubelle Détectée")
        elif pred == "erreur":
            st.error("❌ Erreur de Détection")
        else:
            badge_class = "full-badge" if pred == "pleine" else "empty-badge"
            badge_text = "🗑️ POUBELLE PLEINE" if pred == "pleine" else "🗑️ POUBELLE VIDE"
            st.markdown(f'<div class="{badge_class}">{badge_text}</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2rem; color:#764ba2; text-align:center;'>Score: {score:.1%}</div>", unsafe_allow_html=True)

            # Image annotée
            img_annot = img.copy()
            draw = ImageDraw.Draw(img_annot)
            x, y, w, h = box
            color = "#ff6b6b" if pred == "pleine" else "#1dd1a1"
            for i in range(4):
                draw.rectangle([x-i, y-i, x+w+i, y+h+i], outline=color, width=1)
            st.image(img_annot, caption=f"Poubelle {pred} détectée", use_container_width=True)

elif uploaded_file and not check_model_exists():
    st.error("❌ Impossible de traiter l'image : le modèle n'est pas disponible")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    🚀 Propulsé par YOLOv8 | SmartBin Detector v2.0 | Design Premium
</div>
""", unsafe_allow_html=True)
