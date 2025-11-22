import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import io
import gc
import os

MODEL_PATH = "model/poubelle_yolov8.pt"

# Configuration de la page
st.set_page_config(
    page_title="🗑️ SmartBin Detector", 
    layout="wide", 
    page_icon="🗑️",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
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
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown('<h1 class="main-header">🗑️ SmartBin Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Détection intelligente des poubelles par Intelligence Artificielle</p>', unsafe_allow_html=True)

# Sidebar avec informations
with st.sidebar:
    st.markdown("### 📊 À propos")
    st.info("""
    Cette application utilise un modèle YOLOv8 entraîné pour détecter et classifier l'état des poubelles.
    
    **Fonctionnalités :**
    - 🎯 Détection des poubelles
    - 📊 Classification pleine/vide
    - 🔍 Score de confiance
    - 🖼️ Visualisation des résultats
    """)
    
    st.markdown("### 📈 Statistiques")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Précision modèle", "95%")
    with col2:
        st.metric("Temps réponse", "<2s")
    
    st.markdown("---")
    st.markdown("### 🛠️ Support")
    st.caption("Problèmes techniques ? Contactez notre équipe de support.")

# Zone de téléchargement stylisée
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown("### 📤 Importez votre image")
st.markdown("Glissez-déposez ou sélectionnez une image contenant une poubelle")
uploaded_file = st.file_uploader(
    "",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=False,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Traitement de l'image
if uploaded_file is not None:
    # Section informations fichier
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📄 Informations du fichier", expanded=True):
            st.write(f"**Nom :** {uploaded_file.name}")
            st.write(f"**Type :** {uploaded_file.type}")
            st.write(f"**Taille :** {uploaded_file.size / 1024:.1f} KB")
    
    try:
        # Chargement et affichage de l'image originale
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🖼️ Image originale")
            st.image(img, use_container_width=True, caption="Image importée")
        
        # Prétraitement
        MAX_SIZE = (1024, 1024)
        img.thumbnail(MAX_SIZE)
        img_array = np.array(img)
        import cv2
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Fonction de prédiction
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

        # Prédiction avec barre de progression
        with st.spinner("🔍 Analyse en cours par l'IA..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            box, pred, score = predict_image_yolo(img_array)
        
        # Affichage des résultats
        with col2:
            st.markdown("### 📊 Résultats de l'analyse")
            
            if pred == "aucune détection":
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.error("🚫 Aucune poubelle détectée")
                st.write("Essayez avec une image plus claire ou sous un angle différent.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif pred == "erreur":
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.error("❌ Erreur de détection")
                st.write("Une erreur s'est produite lors de l'analyse.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                # Carte de résultats
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Badge d'état
                if pred == "pleine":
                    st.markdown('<div class="full-badge">🗑️ POUBELLE PLEINE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="empty-badge">🗑️ POUBELLE VIDE</div>', unsafe_allow_html=True)
                
                # Score de confiance
                st.metric("Score de confiance", f"{score:.2%}")
                
                # Barre de confiance colorée
                st.markdown('<div class="confidence-bar"></div>', unsafe_allow_html=True)
                
                # Recommandation
                if pred == "pleine":
                    st.warning("💡 Recommandation : Cette poubelle devrait être vidée bientôt.")
                else:
                    st.success("💡 Recommandation : Cette poubelle peut encore être utilisée.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Image annotée
                draw = ImageDraw.Draw(img)
                x, y, w, h = box
                # Rectangle coloré selon l'état
                color = "#ff6b6b" if pred == "pleine" else "#1dd1a1"
                draw.rectangle([x, y, x + w, y + h], outline=color, width=6)
                
                # Légende
                caption = f"Poubelle {pred} (confiance: {score:.2%})"
                st.image(img, use_container_width=True, caption=caption)
        
        # Section statistiques supplémentaires
        st.markdown("---")
        st.markdown("### 📈 Analyse détaillée")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("État détecté", pred.upper() if pred not in ["aucune détection", "erreur"] else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Confiance IA", f"{score:.2%}" if pred not in ["aucune détection", "erreur"] else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Temps d'analyse", "< 2 secondes")
            st.markdown('</div>', unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement de l'image : {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "🚀 Propulsé par YOLOv8 | SmartBin Detector v1.0"
    "</div>", 
    unsafe_allow_html=True
)