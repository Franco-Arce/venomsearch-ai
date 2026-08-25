from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# Import local modules
from venomsearch.embeddings.esm_encoder import ESMEncoder
from venomsearch.etl.cleaner import SequenceCleaner
from venomsearch.search.vector_store import VenomVectorStore

# Configurar la página de Streamlit
st.set_page_config(
    page_title="VenomSearch-AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados (Glassmorphism / Dark theme)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00d2ff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3a7bd5 0%, #3a6073 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3a6073 0%, #3a7bd5 100%);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Funciones cacheadas
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Carga el modelo ESM-2 y la base de datos vectorial solo una vez."""
    encoder = ESMEncoder(model_name="facebook/esm2_t6_8M_UR50D", device="cpu")
    db_path = Path("data/lancedb")
    store = VenomVectorStore(db_path=db_path)
    cleaner = SequenceCleaner()
    return encoder, store, cleaner

@st.cache_data
def get_all_background_data():
    """Extrae todos los vectores de la DB para entrenar la proyección PCA/UMAP."""
    _, store, _ = load_models()
    try:
        df = store._table.to_pandas()
        return df
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# UI - Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Protein_folding.png/320px-Protein_folding.png", width=200)
    st.title("🧬 Parámetros")

    top_k = st.slider("Resultados a mostrar (Top K)", min_value=1, max_value=20, value=5)

    st.markdown("---")
    st.markdown("""
    ### 🔬 Acerca de
    Este dashboard realiza búsquedas de similitud estructural/funcional usando **ESM-2** (Meta) sobre un dataset curado de toxinas de UniProt.
    
    - **Inferencia:** ESM-2 (8M)
    - **Búsqueda:** LanceDB
    - **Embedding Dim:** 320
    """)

# -----------------------------------------------------------------------------
# UI - Main
# -----------------------------------------------------------------------------
st.title("🐍 VenomSearch-AI: Motor Vectorial de Toxinas")
st.markdown("Descubrí péptidos similares biológicamente mediante Inteligencia Artificial (pLLMs).")

# Cargar recursos
with st.spinner("Cargando Motor ESM-2 y Vector DB..."):
    encoder, store, cleaner = load_models()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Búsqueda de Secuencia")
    # Ejemplo de Melitina de abeja (cardiotoxina/antimicrobiano)
    default_seq = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    query_seq = st.text_area("Ingresá la secuencia de aminoácidos (FASTA o texto plano):", value=default_seq, height=150)

    search_clicked = st.button("🔍 Buscar Similitud")

# -----------------------------------------------------------------------------
# Lógica de Búsqueda
# -----------------------------------------------------------------------------
if search_clicked and query_seq:
    # 1. Limpiar y validar
    query_clean = query_seq.replace("\n", "").replace(" ", "").upper()

    # 2. Generar Embedding
    with st.spinner("Generando Embedding con ESM-2..."):
        emb = encoder.encode_batch([query_clean], show_progress=False)[0]

    # 3. Buscar en LanceDB
    with st.spinner("Buscando en LanceDB..."):
        try:
            results = store.search(emb, top_k=top_k)
        except Exception as e:
            st.error(f"Error en la base de datos (asegurate de haber ejecutado el pipeline CLI antes): {e}")
            results = []

    if results:
        st.success(f"¡Encontrados {len(results)} resultados similares en milisegundos!")

        # Formatear resultados a DF
        res_df = pd.DataFrame(results)
        # La distancia de LanceDB ya fue convertida a cosine_similarity por VenomVectorStore
        res_df["similitud"] = res_df["cosine_similarity"]

        # -----------------------------------------------------------------------------
        # Resultados - Tabla
        # -----------------------------------------------------------------------------
        st.subheader("🏆 Mejores Coincidencias (Top Hits)")
        display_cols = ["accession", "protein_name", "toxin_family", "organism", "similitud"]

        # Mostrar tabla interactiva
        st.dataframe(
            res_df[display_cols].style.background_gradient(subset=["similitud"], cmap="Blues"),
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------------------------------------------
        # Visualización 2D (PCA)
        # -----------------------------------------------------------------------------
        st.subheader("🌌 Espacio Latente (PCA Projection)")
        st.markdown("Dónde se ubica tu secuencia consultada respecto al resto de la base de datos.")

        with st.spinner("Proyectando espacio vectorial..."):
            bg_df = get_all_background_data()
            if not bg_df.empty:
                from sklearn.decomposition import PCA

                # Para velocidad en el dashboard usamos PCA
                pca = PCA(n_components=2)

                # Extraemos vectores de fondo
                bg_vectors = np.stack(bg_df["vector"].values)
                bg_pca = pca.fit_transform(bg_vectors)

                # Transformamos nuestra query
                query_pca = pca.transform(emb.reshape(1, -1))

                plot_df = pd.DataFrame({
                    "x": bg_pca[:, 0],
                    "y": bg_pca[:, 1],
                    "familia": bg_df["toxin_family"],
                    "nombre": bg_df["protein_name"],
                    "tipo": "Dataset (Toxinas)"
                })

                query_plot_df = pd.DataFrame({
                    "x": query_pca[:, 0],
                    "y": query_pca[:, 1],
                    "familia": ["QUERY"],
                    "nombre": ["TU SECUENCIA"],
                    "tipo": "Consulta Actual"
                })

                final_plot_df = pd.concat([plot_df, query_plot_df])

                fig = px.scatter(
                    final_plot_df, x="x", y="y", color="familia",
                    symbol="tipo", hover_data=["nombre"],
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title="Proyección del Espacio Latente ESM-2"
                )

                # Destacar la query
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                # Hacemos la query mucho mas grande
                fig.for_each_trace(lambda t: t.update(marker=dict(size=15, symbol='star', color='yellow', line=dict(width=2, color='DarkSlateGrey'))) if t.name == "QUERY" else ())

                fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------------------------
        # Visualización 3D Molecular (PDBe Molstar)
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔬 Visualización Estructural (AlphaFold 3D)")

        best_match = res_df.iloc[0]
        st.markdown(f"Mostrando predicción de estructura para el mejor resultado: **{best_match['protein_name']}** (`{best_match['accession']}`)")

        # PDBe Molstar Viewer embebido
        molstar_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
          <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/pdbe-molstar@3.1.2/build/pdbe-molstar-light.css">
          <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/pdbe-molstar@3.1.2/build/pdbe-molstar-plugin.js"></script>
          <style>
            #myViewer {{
              width: 100%;
              height: 500px;
              position: relative;
            }}
          </style>
        </head>
        <body>
          <div id="myViewer"></div>
          <script>
            var viewerInstance = new PDBeMolstarPlugin();
            var options = {{
              customData: {{ 
                  url: 'https://alphafold.ebi.ac.uk/files/AF-{best_match['accession']}-F1-model_v4.cif', 
                  format: 'cif' 
              }},
              alphafoldView: true,
              bgColor: {{r: 14, g: 17, b: 23}}, // Match Streamlit dark theme
              hideControls: true
            }};
            var viewerContainer = document.getElementById('myViewer');
            viewerInstance.render(viewerContainer, options);
          </script>
        </body>
        </html>
        """

        components.html(molstar_html, height=520)

    else:
        st.warning("No se encontraron resultados. ¿Ejecutaste el pipeline ETL para llenar la base de datos?")
