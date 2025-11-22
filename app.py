import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO
from ultralytics import YOLO
import gc

st.set_page_config("🗑️ SmartBin Detector", layout="wide")

MODEL_PATH = "model/poubelle_yolov8.pt"

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
        return (x1, y1, x2-x1, y2-y1), label, score
    except Exception as e:
        st.error(f"Erreur YOLO : {e}")
        return None, "erreur", 0.0

st.title("🗑️ SmartBin Detector")
uploaded_file = st.file_uploader("Glissez-déposez une image", type=['jpg','jpeg','png'])

if uploaded_file:
    img = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
    st.image(img, caption="Image importée", use_container_width=True)
    img_array = np.array(img)

    with st.spinner("🔍 Analyse en cours..."):
        box, pred, score = predict_image_yolo(img_array)

    if pred in ["aucune détection","erreur"]:
        st.error(pred)
    else:
        icon = "🟢" if pred=="pleine" else "🔵"
        st.success(f"{icon} Poubelle : {pred} - Confiance : {score:.2%}")
        draw = ImageDraw.Draw(img)
        x,y,w,h = box
        draw.rectangle([x,y,x+w,y+h], outline="yellow", width=4)
        st.image(img, caption="Résultat annoté", use_container_width=True)
