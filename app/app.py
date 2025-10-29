# app.py

import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import os
import logging

# --- Importation des modules locaux ---
from feature_extractor import FeatureExtractor 
from elastic_client import get_elasticsearch_client, search_similar_images
from config import IMAGES_DIR, METRIC_SOURCE, METRIC_NAME, INDEX_NAME, TOP_N_DEFAULT

# =========================================================
# 1. CONFIGURATION ET INITIALISATION DES RESSOURCES
# =========================================================
st.set_page_config(
    page_title="Recherche d'Images par Similarité (Streamlit/Elasticsearch)", 
    layout="wide"
)

@st.cache_resource
def get_feature_extractor():
    """Instancie et met en cache le FeatureExtractor."""
    return FeatureExtractor()

extractor = get_feature_extractor()
es = get_elasticsearch_client() 

# =========================================================
# 2. INTERFACE UTILISATEUR STREAMLIT
# =========================================================

st.title("🖼️ Recherche d'Images par Similarité et Tags (Streamlit)")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Paramètres de Recherche")
    
    top_n = st.number_input("Top K (Nombre de résultats)", min_value=1, max_value=50, value=TOP_N_DEFAULT, step=1)
    
    st.markdown("---")
    
    tag_query = st.text_input("Filtre de Tags/Texte (Optionnel)")

with col2:
    st.subheader("Source de la Requête")
    
    uploaded_file = st.file_uploader(
        "Téléchargez une image (pour la recherche par similarité visuelle)", 
        type=["jpg", "jpeg", "png"]
    )

st.markdown("---")

search_results = []
query_features = np.array([])
search_mode = "Aucun"

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    query_img = Image.open(BytesIO(image_bytes))
    
    st.sidebar.image(query_img, caption="Image de Requête", width=200)

    with st.spinner('Extraction des fonctionnalités vectorielles...'):
        query_features = extractor.extract(query_img)
    
    search_mode = "Hybride (Image + Filtre Textuel)" if tag_query else "Vectorielle Pure"

elif tag_query:
    search_mode = "Textuelle Pure"
    query_features = np.array([]) 

if st.button(f"Lancer la recherche ({search_mode})", type="primary") and (uploaded_file is not None or tag_query):
    
    st.info(f"Démarrage de la recherche en mode: **{search_mode}** (Top {top_n}, Tags: '{tag_query or 'N/A'}')")
    
    if es:
        with st.spinner(f'Recherche dans l\'index "{INDEX_NAME}"...'):
            search_results = search_similar_images(
                es, 
                query_features, 
                METRIC_SOURCE, 
                top_n, 
                tag_query
            ) 
            st.success(f"✅ Recherche terminée. {len(search_results)} résultats trouvés.")
    else:
        st.error("Impossible de lancer la recherche car Elasticsearch n'est pas connecté ou a échoué à l'initialisation.")


st.markdown("## Résultats de la Recherche")

if not search_results:
    st.write("Aucun résultat affiché. Téléchargez une image, ou ajustez les tags/l'image pour correspondre au filtre.")
else:
    num_cols = min(len(search_results), 5)
    cols = st.columns(num_cols) 

    for i, result in enumerate(search_results):
        image_path = os.path.join(IMAGES_DIR, result['relative_path'])
        
        col_index = i % num_cols
        
        with cols[col_index]:
            st.markdown(f"**# {i+1}** (Score: `{result['score']:.4f}`)")
            
            try:
                result_img = Image.open(image_path)
                st.image(result_img, use_container_width=True) 
            except FileNotFoundError:
                st.warning(f"Image non trouvée : {result['relative_path']}")
            except Exception:
                st.warning(f"Erreur d'affichage : {result['relative_path']}")
            
            st.caption(f"**Nom:** {result['image_name']}")
            st.caption(f"**Tags:** `{', '.join(result.get('tags', ['N/A']))}`")