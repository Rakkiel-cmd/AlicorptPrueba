import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from data_gen import cargar_datos_csv

# Descargar NLTK solo la primera vez que arranca el servidor (cacheado)
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

setup_nltk()

st.set_page_config(page_title="NLTK y WordCloud", layout="wide", page_icon="☁️")

# TIPOGRAFÍA Y ESTILOS (consistentes con el resto de la app)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"], button, input, textarea {
        font-family: 'Poppins', sans-serif;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
        <div style="background:#E4572E; border-radius:8px; padding:6px; display:flex;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="12" width="4" height="9" rx="1" fill="white"/>
                <rect x="10" y="7" width="4" height="14" rx="1" fill="white"/>
                <rect x="17" y="3" width="4" height="18" rx="1" fill="white"/>
            </svg>
        </div>
        <span style="font-weight:600; font-size:17px;">Alicorp Analytics</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("☁️ Análisis de sentimiento y nube de palabras.")

st.title("NLTK y WordCloud")
st.write("NLTK analiza el sentimiento de las reseñas y WordCloud arma una nube con las palabras más usadas.")

df_alicorp = cargar_datos_csv()

tab_nltk, tab_wc = st.tabs(["💬 NLTK", "☁️ WordCloud"])

# NLTK
with tab_nltk:
    st.header("NLTK")
    st.write("Cada reseña se clasifica como positiva, negativa o neutra según su contenido. La puntuación compuesta (Compound Score) normaliza el sentimiento entre -1 y +1.")
    st.latex(r"Compound = \frac{\sum \text{valencias}}{\sqrt{\sum \text{valencias}^2 + \alpha}}")
    sia = SentimentIntensityAnalyzer()
    df_alicorp["Sentimiento"] = df_alicorp["Reseña_Cliente"].apply(lambda t: sia.polarity_scores(t)["compound"])
    promedio_sentimiento = df_alicorp["Sentimiento"].mean()
    positivas = (df_alicorp["Sentimiento"] > 0.05).sum()
    negativas = (df_alicorp["Sentimiento"] < -0.05).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Sentimiento Promedio (-1 a 1)", f"{promedio_sentimiento:.2f}")
    c2.metric("Reseñas Positivas", positivas)
    c3.metric("Reseñas Negativas", negativas)


# WORDCLOUD
with tab_wc:
    st.header("WordCloud")
    st.write("La nube resalta las palabras más repetidas en las reseñas. Al agregar una reseña propia, la nube se actualiza al momento.")

    todas_reseñas_originales = " ".join(df_alicorp["Reseña_Cliente"])

    reseña_personalizada = st.text_input("Agrega tu propia reseña aquí (ej. 'Excelente producto, me encanta Blanca Flor'):", "")

    if reseña_personalizada:
        # Para que resalte más, la multiplicamos unas veces
        texto_final = todas_reseñas_originales + (" " + reseña_personalizada) * 50
    else:
        texto_final = todas_reseñas_originales

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_final)

    with st.container(border=True):
        fig_nltk, ax_nltk = plt.subplots(figsize=(8, 4))
        ax_nltk.imshow(wordcloud, interpolation='bilinear')
        ax_nltk.axis('off')
        st.pyplot(fig_nltk)
        plt.close(fig_nltk)
