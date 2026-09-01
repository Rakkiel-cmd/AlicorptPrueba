import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from data_gen import generar_datos_alicorp

# Descargar recursos NLTK si es necesario
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title="NLTK y WordCloud", layout="wide")

st.title("NLTK y Análisis de Sentimiento")
st.write("Procesamiento de Lenguaje Natural sobre las reseñas de clientes.")

df_alicorp = generar_datos_alicorp()

# --- NLTK / WORDCLOUD ---
st.header("7. WordCloud & NLTK: Nube de Palabras Interactiva")
st.write("Agrega una reseña personalizada para ver cómo afecta instantáneamente la nube de palabras.")

todas_reseñas_originales = " ".join(df_alicorp["Reseña_Cliente"])

reseña_personalizada = st.text_input("Agrega tu propia reseña aquí (ej. 'Excelente producto, me encanta Blanca Flor'):", "")

if reseña_personalizada:
    # Para que resalte más, la multiplicamos unas veces
    texto_final = todas_reseñas_originales + (" " + reseña_personalizada) * 50
else:
    texto_final = todas_reseñas_originales

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_final)

fig_nltk, ax_nltk = plt.subplots(figsize=(8, 4))
ax_nltk.imshow(wordcloud, interpolation='bilinear')
ax_nltk.axis('off')
st.pyplot(fig_nltk)
