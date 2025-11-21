import streamlit as st
import os
import gc
import tempfile
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION STREAMLIT
# -------------------------------
st.set_page_config(
    page_title="SmartBin Detector",
    layout="wide",
    page_icon="🗑️"
)

# -------------------------------
# CHEMINS DES MODELES
# -------------------------------
MODEL_PATH = "model/poubelle_yolov8.pt"
DOWNLOAD_MODEL_PATH = "model/poubelle_model.h5"

# -------------------------------
# FONCTION YOLO
# -------------------------------
def predict_image_yolo(img_array):
    try:
        model = YOLO(MODEL_PATH)
        results = model(img_array)  # On passe le np.array directement
        boxes = results[0].boxes

        if len(boxes) == 0:
            del model
            gc.collect()
            return None, "aucune détection", 0.0

        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        label = "pleine" if label_id == 0 else "vide"
        box_tuple = (x1, y1, x2 - x1, y2 - y1)

        del model
        gc.collect()
        return box_tuple, label, score
    except Exception as e:
        st.error(f"Erreur YOLO : {e}")
        return None, "erreur", 0.0

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
st.markdown("### 📤 Upload d'image pour analyse")

uploaded_file = st.file_uploader(
    "Glissez-déposez ou sélectionnez une image",
    type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
)

if uploaded_file:
    try:
        # Ouvrir et convertir en RGB
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
        st.image(img, caption="Image importée", use_column_width=True)

        with st.spinner("🔍 Analyse en cours..."):
            box, pred, score = predict_image_yolo(img_array)

        if pred == "aucune détection":
            st.error("🚫 Aucune poubelle détectée !")
        elif pred == "erreur":
            st.error("❌ Une erreur est survenue lors de la détection.")
        else:
            icon = "🟢" if pred == "pleine" else "🔵"
            st.success(f"### {icon} Poubelle : {pred.capitalize()}\n**Confiance : {score:.2%}**")

            # Dessiner la box sur l'image
            draw = ImageDraw.Draw(img)
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline="yellow", width=4)
            st.image(img, caption="Résultat annoté", use_column_width=True)

    except Exception as e:
        st.error(f"Erreur traitement image : {e}")
