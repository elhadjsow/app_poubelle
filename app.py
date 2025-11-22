import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2
import gc
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "model/poubelle_yolov8.pt"

st.set_page_config(
    page_title="🗑️ SmartBin Detector", 
    layout="wide", 
    page_icon="🗑️",
    initial_sidebar_state="expanded"
)

# -------------------------------
# FONCTIONS
# -------------------------------
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
# SIDEBAR : Gestion du modèle
# -------------------------------
with st.sidebar:
    st.header("🛠️ Gestion du Modèle")
    
    if check_model_exists():
        st.success("✅ Modèle disponible")
        # Bouton pour télécharger le modèle
        with open(MODEL_PATH, "rb") as f:
            st.download_button(
                label="📥 Télécharger le modèle",
                data=f,
                file_name="poubelle_yolov8.pt",
                use_container_width=True
            )
    else:
        st.error("❌ Modèle non trouvé ! Placez 'poubelle_yolov8.pt' dans le dossier 'model'")

# -------------------------------
# HEADER PRINCIPAL
# -------------------------------
st.title("🗑️ SmartBin Detector")
st.write("Détection intelligente des poubelles par Intelligence Artificielle")

# -------------------------------
# UPLOAD ZONE
# -------------------------------
if not check_model_exists():
    st.warning("⏳ Veuillez d'abord télécharger le modèle pour activer la détection")
    uploaded_file = None
else:
    uploaded_file = st.file_uploader("📤 Importez votre image", type=['jpg','jpeg','png'])

# -------------------------------
# TRAITEMENT ET AFFICHAGE
# -------------------------------
if uploaded_file and check_model_exists():
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Image originale", use_container_width=True)
    
    with st.spinner("🔍 Analyse en cours..."):
        box, pred, score = predict_image_yolo(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    
    with col2:
        st.subheader("Résultat")
        if pred == "aucune détection":
            st.error("🚫 Aucune poubelle détectée")
        elif pred == "erreur":
            st.error("❌ Erreur lors de l'analyse")
        else:
            badge_text = "🗑️ POUBELLE PLEINE" if pred == "pleine" else "🗑️ POUBELLE VIDE"
            st.success(f"{badge_text} - Confiance: {score:.1%}")
            
            # Affichage image annotée
            img_annot = img.copy()
            draw = ImageDraw.Draw(img_annot)
            x, y, w, h = box
            color = "#ff6b6b" if pred == "pleine" else "#1dd1a1"
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            st.image(img_annot, caption="Poubelle détectée", use_container_width=True)

elif uploaded_file and not check_model_exists():
    st.error("❌ Impossible de traiter l'image : le modèle n'est pas disponible")
