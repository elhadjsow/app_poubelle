import streamlit as st
import os
import gc
import tempfile
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
# CSS GLOBAL — DESIGN MODERNE PREMIUM
# -------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    body, html, * {
        font-family: 'Inter', sans-serif !important;
    }

    /* Header principal */
    .header-container {
        background: linear-gradient(135deg, #4b6cb7, #182848);
        padding: 3rem 1rem;
        border-radius: 0 0 40px 40px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 35px rgba(0,0,0,0.2);
    }
    .header-title { font-size: 3.8rem; font-weight: 700; letter-spacing: -1px; }
    .header-sub { font-size: 1.2rem; opacity: 0.9; }

    /* Zone upload */
    .upload-zone {
        border: 3px dashed #4b6cb7;
        padding: 2.5rem;
        border-radius: 20px;
        background: #f5f7ff;
        transition: 0.3s;
        text-align: center;
    }
    .upload-zone:hover {
        background: #e9ecff;
        border-color: #182848;
    }

    /* Cartes */
    .card {
        background: white;
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    /* Résultat */
    .result-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
    }

    /* Boutons */
    .styled-btn {
        background: linear-gradient(135deg, #4b6cb7, #182848);
        color: white !important;
        padding: 1rem 1.8rem;
        border-radius: 50px;
        font-size: 1.1rem;
        border: none;
        width: 100%;
        margin-top: 0.5rem;
        transition: 0.3s;
        box-shadow: 0 5px 20px rgba(75,108,183,0.4);
    }
    .styled-btn:hover {
        background: linear-gradient(135deg, #5c7ed5, #203060);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(75,108,183,0.6);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# MODELES
# -------------------------------
MODEL_PATH = "model/poubelle_yolov8.pt"
DOWNLOAD_MODEL_PATH = "model/poubelle_model.h5"

# -------------------------------
# Fonction YOLO
# -------------------------------
def predict_image_yolo(img_path):
    try:
        model = YOLO(MODEL_PATH)
        results = model(img_path)
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
# HEADER
# -------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">🗑️ SmartBin Detector</div>
    <p class="header-sub">Détection intelligente des poubelles avec IA • YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# CONTENU PRINCIPAL
# -------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload d'image pour analyse")
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Glissez-déposez ou sélectionnez une image",
        type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    )

    st.markdown('</div></div>', unsafe_allow_html=True)

    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Image importée", use_column_width=True)

            # Utilisation d'un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                img_path = tmp_file.name
                img.save(img_path)

            with st.spinner("🔍 Analyse en cours..."):
                box, pred, score = predict_image_yolo(img_path)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            if pred == "aucune détection":
                st.error("🚫 Aucune poubelle détectée !")
            elif pred == "erreur":
                st.error("❌ Une erreur est survenue lors de la détection.")
            else:
                icon = "🟢" if pred == "pleine" else "🔵"
                st.success(f"### {icon} Poubelle : {pred.capitalize()}\n**Confiance : {score:.2%}**")

                # Draw box
                draw = ImageDraw.Draw(img)
                x, y, w, h = box
                draw.rectangle([x, y, x + w, y + h], outline="yellow", width=4)
                st.image(img, caption="Résultat annoté", use_column_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur traitement image : {e}")

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📥 Télécharger le modèle YOLO")

    if os.path.exists(DOWNLOAD_MODEL_PATH):
        with open(DOWNLOAD_MODEL_PATH, "rb") as f:
            st.download_button(
                "📦 Télécharger le modèle",
                data=f,
                file_name="poubelle_model.h5"
            )
    else:
        st.error("Fichier du modèle introuvable.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class="footer">
    🌍 SmartBin Detector — Développé avec Streamlit & YOLOv8  
</div>
""", unsafe_allow_html=True)
