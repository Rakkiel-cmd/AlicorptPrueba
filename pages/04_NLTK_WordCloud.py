import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
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
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FDF6F2 100%);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    @keyframes logoGlowPulse {
        0% { box-shadow: 0 0 0 0 rgba(228,87,46,0.55); }
        70% { box-shadow: 0 0 0 14px rgba(228,87,46,0); }
        100% { box-shadow: 0 0 0 0 rgba(228,87,46,0); }
    }
    .logo-glow {
        animation: logoGlowPulse 2.2s infinite;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
        <div class="logo-glow" style="background:#E4572E; border-radius:8px; padding:6px; display:flex;">
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

    # VADER solo entiende inglés. Si las reseñas están en español, el compound
    # sale mal (por ejemplo, "no" se detecta como negación y sesga todo a negativo).
    # Por eso traducimos cada reseña al inglés antes de analizarla.
    @st.cache_data
    def traducir_reseñas(reseñas):
        traductor = GoogleTranslator(source="es", target="en")
        return [traductor.translate(r) for r in reseñas]

    with st.spinner("Analizando sentimiento de las reseñas..."):
        reseñas_en = traducir_reseñas(df_alicorp["Reseña_Cliente"].tolist())
        sia = SentimentIntensityAnalyzer()
        df_alicorp["Sentimiento"] = [sia.polarity_scores(t)["compound"] for t in reseñas_en]

    promedio_sentimiento = df_alicorp["Sentimiento"].mean()
    positivas = (df_alicorp["Sentimiento"] > 0.05).sum()
    negativas = (df_alicorp["Sentimiento"] < -0.05).sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("Sentimiento Promedio (-1 a 1)", f"{promedio_sentimiento:.2f}")
    with c2:
        with st.container(border=True):
            st.metric("Reseñas Positivas", positivas)
    with c3:
        with st.container(border=True):
            st.metric("Reseñas Negativas", negativas)

    with st.expander("📋 Ver detalle por reseña"):
        st.dataframe(
            df_alicorp[["Reseña_Cliente", "Sentimiento"]],
            use_container_width=True,
            hide_index=True,
        )


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

    # WordCloud usa stopwords en inglés por defecto. Sin esto, palabras como
    # "que", "de", "la", "el" saldrían gigantes en la nube sin aportar nada.
    stopwords_es = set(STOPWORDS)
    stopwords_es.update([
        "que", "de", "la", "el", "en", "y", "es", "un", "una", "los", "las",
        "por", "con", "para", "muy", "se", "su", "lo", "al", "del", "mi",
        "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque",
    ])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords_es,
        colormap='autumn',
    ).generate(texto_final)

    with st.container(border=True):
        fig_nltk, ax_nltk = plt.subplots(figsize=(8, 4))
        ax_nltk.imshow(wordcloud, interpolation='bilinear')
        ax_nltk.axis('off')
        st.pyplot(fig_nltk)
        plt.close(fig_nltk)
