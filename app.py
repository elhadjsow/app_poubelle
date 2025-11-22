import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import io
import gc

MODEL_PATH = "model/poubelle_yolov8.pt"

st.set_page_config(page_title="🗑️ SmartBin Detector", layout="wide", page_icon="🗑️")

st.markdown("## 🗑️ SmartBin Detector - Détection des poubelles avec IA")

# ------------------------------
# Upload d'image
# ------------------------------
uploaded_file = st.file_uploader(
    "📤 Glissez-déposez ou sélectionnez une image",
    type=['jpg', 'jpeg', 'png']
)

def predict_image_yolo(img_array):
    try:
        # Charger le modèle YOLO
        model = YOLO(MODEL_PATH)
        results = model(img_array)
        boxes = results[0].boxes

        if len(boxes) == 0:
            del model; gc.collect()
            return None, "aucune détection", 0.0

        # On prend la première box
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

if uploaded_file:
    try:
        # Repositionner le curseur du fichier
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")

        # Limiter la taille
        MAX_SIZE = (1024, 1024)
        img.thumbnail(MAX_SIZE)

        # Convertir en numpy array (BGR pour YOLO si nécessaire)
        img_array = np.array(img)
        import cv2
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.image(img, caption="Image importée", use_container_width=True)

        with st.spinner("🔍 Analyse en cours..."):
            box, pred, score = predict_image_yolo(img_array)

        if pred == "aucune détection":
            st.warning("🚫 Aucune poubelle détectée !")
        elif pred == "erreur":
            st.error("❌ Une erreur est survenue lors de la détection.")
        else:
            icon = "🟢" if pred == "pleine" else "🔵"
            st.success(f"{icon} Poubelle détectée : {pred.capitalize()} (Confiance : {score:.2%})")

            # Dessiner la box sur l'image
            draw = ImageDraw.Draw(img)
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline="yellow", width=4)

            st.image(img, caption="Résultat annoté", use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image : {e}")
