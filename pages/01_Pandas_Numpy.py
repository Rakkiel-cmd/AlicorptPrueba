import streamlit as st
import numpy as np
from data_gen import generar_datos_alicorp

st.set_page_config(page_title="Pandas y NumPy", layout="wide")

st.title("🧩 Integrante 1: Pandas y NumPy")
st.write("Análisis y limpieza de datos, y transformaciones matemáticas.")

df_alicorp = generar_datos_alicorp()

# --- PANDAS ---
st.header("1. Pandas: Manejo de Datos (Interactivo)")
st.write("Filtra el DataFrame por categoría de producto para ver los resultados en vivo.")

# Interactividad: Selectbox para filtrar
categorias_disponibles = ["Todas"] + list(df_alicorp["Categoria"].unique())
categoria_seleccionada = st.selectbox("Selecciona una Categoría para Filtrar:", categorias_disponibles)

if categoria_seleccionada == "Todas":
    df_filtrado = df_alicorp
else:
    df_filtrado = df_alicorp[df_alicorp["Categoria"] == categoria_seleccionada]

resumen_pandas = df_filtrado.groupby("Marca")["Ventas_Soles"].sum().reset_index().sort_values(by="Ventas_Soles", ascending=False)
st.dataframe(resumen_pandas.style.highlight_max(subset=["Ventas_Soles"], color="lightgreen"), width="stretch")

st.markdown("---")

# --- NUMPY ---
st.header("2. NumPy: Transformaciones Matriciales (Interactivo)")
st.write("Genera una nueva matriz de factores logísticos alterando el multiplicador.")

# Interactividad: Slider
multiplicador = st.slider("Multiplicador Logístico", min_value=1.0, max_value=20.0, value=10.0, step=1.0)

matriz_costos_2x2 = np.array([[120.5, 45.2], [80.0, 110.3]])
matriz_transformacion_3x3 = np.random.rand(3,3) * multiplicador

col_np1, col_np2 = st.columns(2)
col_np1.write("**Matriz de Costos Base (2x2):**")
col_np1.write(matriz_costos_2x2)

col_np2.write(f"**Matriz de Factores Logísticos (3x3) (x {multiplicador}):**")
col_np2.write(matriz_transformacion_3x3)
