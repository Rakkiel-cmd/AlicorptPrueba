# Alicorp Analytics & ML

Proyecto académico (SENATI) que demuestra, en una app de **Streamlit**
con varias páginas, el uso práctico de las principales librerías de análisis
de datos y Machine Learning en Python. Los nombres de marca y categorías corresponden
a productos reales de Alicorp, pero los registros de ventas, ganancias, reseñas y
fechas son **simulados por código** (`alicorp_simulated_data.csv`, 1000 registros) y
no corresponden a datos reales de la empresa.

**App en vivo:** https://alicorptprueba-9mdxpnekpu7ccfrknfcyyg.streamlit.app

## ¿De qué trata?

La app tiene un dashboard principal y una página por cada librería, todas
usando el mismo dataset simulado (ventas, ganancias, tiempos de entrega, reseñas de clientes, etc.):

| Página | Librería(s) | Qué muestra |
|---|---|---|
| `app.py` | Streamlit, Pandas | Dashboard con KPIs (ventas, ganancias, tiempo de entrega) y evolución mensual de ventas |
| `01_Pandas_Numpy.py` | Pandas, NumPy | Filtrado/agrupación de ventas por marca y categoría; manejo de matrices |
| `02_Matplotlib_Seaborn.py` | Matplotlib, Seaborn | Gráficos comparativos de ventas y boxplots de distribución (IQR) |
| `03_ScikitLearn_SciPy.py` | Scikit-learn, SciPy | Clasificación con Random Forest (matriz de confusión) y ajuste de curva normal |
| `04_NLTK_WordCloud.py` | NLTK, WordCloud | Análisis de sentimiento de reseñas y nube de palabras interactiva |
| `05_TensorFlow_Keras.py` | TensorFlow, Keras | Arquitectura de red neuronal, heatmap de pesos y superficie de pérdida 3D |
| `06_PyTorch.py` | PyTorch | Entrenamiento real de un modelo de regresión lineal (ventas → ganancias) con Momentum |

## Qué resuelve

Sirve como portafolio/demo para mostrar, con datos y gráficos interactivos
, cómo se usa cada librería en un caso de negocio simulado (analítica comercial + Machine Learning),
en vez de ejemplos genéricos aislados.

## Requisitos cumplidos (proyecto final de Machine Learning)

| Requisito pedido | Dónde se cumple | Estado |
|---|---|---|
| CSV | `alicorp_simulated_data.csv` (1000 registros simulados) | ✅ |
| Página web | App completa en Streamlit, desplegada en Streamlit Cloud | ✅ |
| Python | Todo el proyecto | ✅ |
| Pandas | `data_gen.py` y todas las páginas | ✅ |
| Matplotlib | `pages/02_Matplotlib_Seaborn.py`, `03`, `05`, `06` | ✅ |
| Redes neuronales | `pages/05_TensorFlow_Keras.py` (Keras) y `pages/06_PyTorch.py` (PyTorch) | ✅ |

Adicional al mínimo pedido: NumPy, Seaborn, Scikit-learn, SciPy, NLTK y WordCloud.

## Cómo correrlo localmente

```bash
python -m venv venv
source venv/bin/activate    # En Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Estructura del proyecto

```
├── app.py                      # Dashboard principal
├── data_gen.py                 # Carga el CSV de datos simulados
├── alicorp_simulated_data.csv  # Dataset simulado (1000 filas)
├── pages/                      # Una página por librería (ver tabla arriba)
├── requirements.txt            # Dependencias
├── nltk.txt                    # Recurso de NLTK a descargar en Streamlit Cloud
└── .devcontainer/               # Configuración para GitHub Codespaces
```

## Estado conocido / pendientes

- ✅ Corregido: los KPIs del dashboard ahora comparan una ventana móvil
  de 30 días contra los 30 días previos (antes comparaba mes calendario contra mes
  calendario, lo que daba caídas falsas cuando el mes en curso tenía muy pocos días de datos).
- ⚠️ Pendiente: fijar versiones exactas en `requirements.txt` y evitar aplicar el
  índice de PyTorch a todo el archivo, para builds más rápidos y estables en Streamlit Cloud.
- ⚠️ Pendiente: manejo de errores si el CSV cambia de formato o faltan columnas.
