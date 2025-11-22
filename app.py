import os
import gc
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
import streamlit as st
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION STREAMLIT
# -------------------------------
st.set_page_config(
    page_title="🗑️ SmartBin Detector",
    layout="wide",
    page_icon="🗑️"
)

MODEL_PATH = "model/poubelle_yolov8.pt"

# -------------------------------
# FONCTION YOLO
# -------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

def predict_image_yolo(img_array):
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
# HEADER
# -------------------------------
st.title("🗑️ SmartBin Detector")
st.write("Détection intelligente des poubelles avec YOLOv8")

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
uploaded_file = st.file_uploader(
    "Glissez-déposez ou sélectionnez une image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    img = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
    st.image(img, caption="Image importée", use_container_width=True)

    img_array = np.array(img)

    with st.spinner("🔍 Analyse en cours..."):
        box, pred, score = predict_image_yolo(img_array)

    if pred == "aucune détection":
        st.error("🚫 Aucune poubelle détectée !")
    elif pred == "erreur":
        st.error("❌ Une erreur est survenue.")
    else:
        icon = "🟢" if pred == "pleine" else "🔵"
        st.success(f"{icon} Poubelle : {pred.capitalize()} - Confiance : {score:.2%}")

        # Dessiner la box sur l'image
        draw = ImageDraw.Draw(img)
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline="yellow", width=4)
        st.image(img, caption="Résultat annoté", use_container_width=True)
