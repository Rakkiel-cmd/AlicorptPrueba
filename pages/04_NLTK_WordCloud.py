import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
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

#configura la ventana del navegador 
st.set_page_config(page_title="NLTK y WordCloud", layout="wide", page_icon="☁️")

# TIPOGRAFÍA Y ESTILOS y ponemos eso xq podemos usar cs y html y al mismo tiemnpo textos normales
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
   
    </style>
    """,
    unsafe_allow_html=True,#le da permiso a streamlit para aceptar estilos visuales personalizados
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
    """,#diseño de la barra lateral
    unsafe_allow_html=True,#le da permiso a streamlit para aceptar estilos visuales personalizados
)
st.sidebar.caption("☁️ Análisis de sentimiento y nube de palabras.")
#pie de pagina en la barra lateral 

st.title("NLTK y WordCloud")#titulo primncipal 
st.write("NLTK analiza el sentimiento de las reseñas y WordCloud arma una nube con las palabras más usadas.")

df_alicorp = cargar_datos_csv()#trae los datos  lo guarda en esa funcion

tab_nltk, tab_wc = st.tabs(["💬 NLTK", "☁️ WordCloud"])

# NLTK
with tab_nltk: #todo lo que pongamos aqui solo aparecera en nlyk
    st.header("NLTK") #titulo
    st.write("Cada reseña se clasifica como positiva, negativa o neutra según su contenido. La puntuación compuesta (Compound Score) normaliza el sentimiento entre -1 y +1.")
    st.latex(r"Compound = \frac{\sum \text{valencias}}{\sqrt{\sum \text{valencias}^2 + \alpha}}")
    #funcion que dibuja formula matematica real que utiliza VADER para sumar valencas ; para que el resultado de entre -1 y +1
   
  
    # Le agregamos un lexicón de palabras en español ya que NLTK esta entrenada en ingles
    ## Diccionario en español: le asigna un puntaje (+ o -) a cada palabra para que NLTK entienda nuestro idioma.
    LEXICON_ES = {
        "excelente": 3.5, "bueno": 2.0, "buena": 2.0, "genial": 3.0,
        "encanta": 3.0, "encantó": 3.0, "increíble": 3.0, "perfecto": 3.0,
        "recomendado": 2.5, "satisfecho": 2.0, "feliz": 2.5, "delicioso": 2.5,
        "rico": 2.0, "calidad": 1.5, "rápido": 1.5, "amable": 2.0,
        "malo": -3.0, "mala": -3.0, "pésimo": -3.5, "terrible": -3.5,
        "horrible": -3.5, "decepcionado": -2.5, "decepcionante": -2.5,
        "lento": -1.5, "caro": -1.0, "defectuoso": -3.0, "roto": -2.5,
        "queja": -2.0, "problema": -1.5, "nunca": -1.0, "tarde": -1.0,
    }
    # Inicializamos el motor de análisis de sentimientos de NLTK
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(LEXICON_ES) #le agregamos el diccionario que creamos en el motor de sentimientos
    df_alicorp["Sentimiento"] = df_alicorp["Reseña_Cliente"].apply(lambda t: sia.polarity_scores(t)["compound"])
    #recore la columna de reseñas y aplica el ,motor para extraer el puntaje compuesto y lo guarda en esa nueva columna
    promedio_sentimiento = df_alicorp["Sentimiento"].mean()
    positivas = (df_alicorp["Sentimiento"] > 0.05).sum()
    negativas = (df_alicorp["Sentimiento"] < -0.05).sum()

    c1, c2, c3 = st.columns(3)#divide la panatlla en 3 columnas 
    c1.metric("Sentimiento Promedio (-1 a 1)", f"{promedio_sentimiento:.2f}")
    c2.metric("Reseñas Positivas", positivas)
    c3.metric("Reseñas Negativas", negativas)


# WORDCLOUD
with tab_wc:
    st.header("WordCloud")
    st.write("La nube resalta las palabras más repetidas en las reseñas. Al agregar una reseña propia, la nube se actualiza al momento.")

    #junta todas las reseñas de los clientes en un solo texto ya que asi lo lee esa libreria
    todas_reseñas_originales = " ".join(df_alicorp["Reseña_Cliente"])
    
    #usuario escriba su propia opinion
    reseña_personalizada = st.text_input("Agrega tu propia reseña aquí (ej. 'Excelente producto, me encanta Blanca Flor'):", "")

    if reseña_personalizada:
        #al escribir algo nuevo tiene que destacar asi que se multiplica por 50
        texto_final = todas_reseñas_originales + (" " + reseña_personalizada) * 50
    else:
        texto_final = todas_reseñas_originales


    stopwords_es = set(STOPWORDS)#copia la lista de palabras prohibidas que ya trae la libreria 
    #le agregamos nuestra popia lista
    stopwords_es.update([
        "que", "de", "la", "el", "en", "y", "es", "un", "una", "los", "las",
        "por", "con", "para", "muy", "se", "su", "lo", "al", "del",
    ])

    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_es).generate(texto_final)

    with st.container(border=True):
        fig_nltk, ax_nltk = plt.subplots(figsize=(8, 4))#prepara el cuadro xon el tamaño correcto
        ax_nltk.imshow(wordcloud, interpolation='bilinear')#dibuja la linea de palabras ahi dentro
        ax_nltk.axis('off')#para ocultar los ejes cartesianos del gráfico de Matplotlib
        st.pyplot(fig_nltk)#muestra en la pantalla ese cuadro
        plt.close(fig_nltk)#cierra el grafico en la memoria de la computadora 
