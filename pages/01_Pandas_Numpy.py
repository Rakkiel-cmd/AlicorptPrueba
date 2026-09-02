# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import numpy as np
from data_gen import generar_datos_alicorp

st.set_page_config(page_title="Pandas y NumPy", layout="wide")

st.title("Pandas y NumPy")
st.write("Pandas se usa para organizar y filtrar los datos, y NumPy para trabajar con matrices numéricas.")

df_alicorp = generar_datos_alicorp()

# PANDAS
st.header("Pandas")
st.write("La tabla agrupa las ventas por marca. El filtro permite limitarla a una sola categoría de producto.")

# Interactividad: Selectbox para filtrar
categorias_disponibles = ["Todas"] + list(df_alicorp["Categoria"].unique())
categoria_seleccionada = st.selectbox("Selecciona una Categoría para Filtrar:", categorias_disponibles)

if categoria_seleccionada == "Todas":
    df_filtrado = df_alicorp
else:
    df_filtrado = df_alicorp[df_alicorp["Categoria"] == categoria_seleccionada]

resumen_pandas = df_filtrado.groupby("Marca")["Ventas_Soles"].sum().reset_index().sort_values(by="Ventas_Soles", ascending=False)
st.dataframe(resumen_pandas.style.highlight_max(subset=["Ventas_Soles"], color="lightgreen"), width="stretch")


# NUMPY
st.header("NumPy")
st.write("Las matrices representan costos y factores logísticos. El multiplicador cambia los valores de la matriz 3x3.")

# Interactividad: Slider
multiplicador = st.slider("Multiplicador Logístico", min_value=1.0, max_value=20.0, value=10.0, step=1.0)

matriz_costos_2x2 = np.array([[120.5, 45.2], [80.0, 110.3]])
# 3x3 aleatoria: representa cómo NumPy maneja arreglos multidimensionales, no solo tablas
matriz_transformacion_3x3 = np.random.rand(3,3) * multiplicador

col_np1, col_np2 = st.columns(2)
col_np1.write("**Matriz de Costos Base (2x2):**")
col_np1.write(matriz_costos_2x2)

col_np2.write(f"**Matriz de Factores Logísticos (3x3) (x {multiplicador}):**")
col_np2.write(matriz_transformacion_3x3)
