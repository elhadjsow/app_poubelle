import os
import gc
import tempfile
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
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
# CSS GLOBAL — DESIGN MODERNE PREMIUM
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

body, html, * { font-family: 'Inter', sans-serif !important; }

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

.upload-zone {
    border: 3px dashed #4b6cb7;
    padding: 3rem;
    border-radius: 20px;
    background: #f5f7ff;
    transition: 0.3s;
    text-align: center;
    margin: 1rem 0;
}
.upload-zone:hover { background: #e9ecff; border-color: #182848; }

.upload-text {
    font-size: 1.3rem;
    font-weight: 600;
    color: #4b6cb7;
    margin-bottom: 1rem;
}

.upload-subtext {
    font-size: 1rem;
    color: #666;
    margin-bottom: 1.5rem;
}

.card {
    background: white;
    padding: 1.8rem;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

.result-box {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

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

.footer { text-align: center; padding: 2rem; color: #777; }

/* Style personnalisé pour le file_uploader */
div[data-testid="stFileUploader"] {
    width: 100%;
}

div[data-testid="stFileUploader"] section {
    padding: 2rem;
    border: 3px dashed #4b6cb7;
    border-radius: 15px;
    background: #f8faff;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: #182848;
    background: #eef2ff;
}

/* Cacher le file_uploader par défaut et utiliser notre style */
.hidden-uploader {
    opacity: 0;
    position: absolute;
    z-index: -1;
}

.custom-upload-label {
    display: inline-block;
    background: linear-gradient(135deg, #4b6cb7, #182848);
    color: white;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: 0.3s;
    box-shadow: 0 5px 20px rgba(75,108,183,0.4);
    margin: 1rem 0;
}

.custom-upload-label:hover {
    background: linear-gradient(135deg, #5c7ed5, #203060);
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(75,108,183,0.6);
}
</style>
""", unsafe_allow_html=True)

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
# FONCTION YOLO
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
# CONTENU PRINCIPAL
# -------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload d'image pour analyse")
    
    # Zone d'upload personnalisée
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown('<div class="upload-text">📁 Glissez-déposez votre image ici</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-subtext">Limite : 200MB par fichier • JPG, JPEG, PNG</div>', unsafe_allow_html=True)
    
    # File uploader avec style personnalisé
    uploaded_file = st.file_uploader(
        "Choisir un fichier",
        type=['jpg','jpeg','png','JPG','JPEG','PNG'],
        key="file_uploader",
        label_visibility="collapsed"  # On cache le label par défaut
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # Afficher les informations du fichier
            st.success(f"✅ Fichier sélectionné : **{uploaded_file.name}**")
            
            # Lire et afficher l'image
            image = Image.open(uploaded_file).convert("RGB")
            
            st.subheader("🖼️ Image importée")
            st.image(image, use_column_width=True)
            
            # Sauvegarder dans un fichier temporaire pour YOLO
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name, format="JPEG", quality=95)
                img_path = tmp_file.name

            # Analyse avec YOLO
            with st.spinner("🔍 Analyse YOLO en cours... Veuillez patienter."):
                box, pred, score = predict_image_yolo(img_path)

            # Affichage des résultats
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("📊 Résultats de l'analyse")
            
            if pred == "aucune détection":
                st.error("🚫 Aucune poubelle détectée dans l'image !")
                st.info("💡 Essayez avec une image où une poubelle est plus visible.")
            elif pred == "erreur":
                st.error("❌ Erreur lors de l'analyse. Vérifiez votre modèle YOLO.")
            else:
                icon = "🗑️🟢" if pred == "pleine" else "🗑️🔵"
                st.success(f"### {icon} État détecté : **{pred.upper()}**")
                st.info(f"**Niveau de confiance : {score:.1%}**")

                # Annoter l'image si une boîte a été détectée
                if box:
                    st.subheader("🎯 Détection visuelle")
                    annotated_image = image.copy()
                    draw = ImageDraw.Draw(annotated_image)
                    x, y, w, h = box
                    
                    # Dessiner le rectangle de détection
                    draw.rectangle([x, y, x + w, y + h], outline="yellow", width=6)
                    
                    # Ajouter le label avec fond
                    label_text = f"{pred} ({score:.1%})"
                    text_bbox = draw.textbbox((x, y-30), label_text)
                    draw.rectangle(text_bbox, fill="yellow")
                    draw.text((x, y-30), label_text, fill="black", font=None)
                    
                    st.image(annotated_image, use_column_width=True, caption="Image avec détection")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Nettoyer le fichier temporaire
            try:
                os.unlink(img_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement : {str(e)}")
            st.info("🔧 Vérifiez que l'image n'est pas corrompue et réessayez.")

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📥 Télécharger le modèle")
    
    if os.path.exists(DOWNLOAD_MODEL_PATH):
        st.success("✅ Modèle disponible au téléchargement")
        with open(DOWNLOAD_MODEL_PATH, "rb") as f:
            st.download_button(
                "🗂️ Télécharger poubelle_model.h5", 
                data=f, 
                file_name="poubelle_model.h5",
                use_container_width=True,
                key="download_model"
            )
    else:
        st.warning("⚠️ Modèle non trouvé")
        st.info("Le fichier model/poubelle_model.h5 n'existe pas encore.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Comment utiliser")
    st.markdown("""
    1. **Glissez-déposez** une image dans la zone ci-contre
    2. **Ou cliquez** pour parcourir vos fichiers
    3. **Attendez** l'analyse automatique
    4. **Visualisez** les résultats de détection
    
    **Formats supportés :** JPG, JPEG, PNG
    **Taille max :** 200MB
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Informations techniques
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔧 Informations techniques")
    st.markdown("""
    - **Modèle :** YOLOv8
    - **Tâche :** Détection d'objets
    - **Classes :** Poubelle pleine/vide
    - **Framework :** Streamlit + Ultralytics
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class="footer">
    🌍 SmartBin Detector — Développé avec Streamlit & YOLOv8
</div>
""", unsafe_allow_html=True)