import streamlit as st
import os
import gc
from PIL import Image
from ultralytics import YOLO

# -------------------------------
# Configuration Streamlit
# -------------------------------
st.set_page_config(
    page_title="Détection Intelligente de Poubelles",
    layout="centered",
)

# -------------------------------
# CSS Style
# -------------------------------
st.markdown("""
<style>
.upload-container {
    background: #ffffff;
    padding: 30px;
    border-radius: 20px;
    border: 2px dashed #667eea;
    text-align: center;
    transition: 0.3s;
}
.upload-container:hover {
    background: #e8eaff;
    border-color: #5a6fe3;
}

/* Bouton Streamlit */
.stButton>button {
    background: #667eea;
    color: white;
    border-radius: 10px;
    padding: 0.7rem 1.5rem;
    border: none;
}
.stButton>button:hover {
    background: #5568d9;
}
</style>
""", unsafe_allow_html=True)

st.title("🗑️ Détection Intelligente : Poubelle Pleine ou Vide")
st.write("Téléversez une image pour analyser l'état de la poubelle.")

# -------------------------------
# Chargement du modèle YOLO
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # mets ici ton modèle entraîné

model = load_model()


# -------------------------------
# Zone d'Upload avec drag & drop
# -------------------------------
with st.container():
    st.markdown("""
    <div class="upload-container">
        <h3 style='color: #667eea;'>📤 Importez votre image</h3>
        <p style='color: #6c757d;'>Glissez-déposez ou sélectionnez une image contenant une poubelle</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Importer une image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# Analyse + Affichage
# -------------------------------
if uploaded_file is not None:
    st.subheader("🖼️ Image importée")
    img = Image.open(uploaded_file)
    st.image(img, width=350)

    st.subheader("🔍 Analyse en cours...")

    results = model.predict(img, conf=0.5)

    # Récupérer les résultats
    boxes = results[0].boxes
    annotated_img = results[0].plot()

    # Affichage image annotée
    st.subheader("📌 Résultat")
    st.image(annotated_img, caption="Détection YOLO", use_column_width=True)

    # Message poubelle pleine/vide (selon ton entraînement)
    if len(boxes) == 0:
        st.warning("⚠️ Aucune poubelle détectée.")
    else:
        classes = results[0].names
        detected_classes = [classes[int(c)] for c in boxes.cls]

        if "poubelle_pleine" in detected_classes:
            st.error("🟥 La poubelle est **pleine** !")
        elif "poubelle_vide" in detected_classes:
            st.success("🟩 La poubelle est **vide**.")
        else:
            st.info("ℹ️ Poubelle détectée, mais classe inconnue.")

    gc.collect()

