import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from data_gen import generar_datos_alicorp

# Descargar NLTK solo la primera vez que arranca el servidor (cacheado)
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

setup_nltk()

st.set_page_config(page_title="NLTK y WordCloud", layout="wide")

st.title("NLTK y WordCloud")
st.write("NLTK analiza el sentimiento de las reseñas y WordCloud arma una nube con las palabras más usadas.")

df_alicorp = generar_datos_alicorp()

# NLTK
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

fig_nltk, ax_nltk = plt.subplots(figsize=(8, 4))
ax_nltk.imshow(wordcloud, interpolation='bilinear')
ax_nltk.axis('off')
st.pyplot(fig_nltk)
plt.close(fig_nltk)
