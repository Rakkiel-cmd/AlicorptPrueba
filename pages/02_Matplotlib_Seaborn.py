import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_gen import generar_datos_alicorp

st.set_page_config(page_title="Matplotlib y Seaborn", layout="wide")

st.title("🧩 Integrante 2: Matplotlib y Seaborn")
st.write("Visualización de tendencias y análisis de distribuciones.")

df_alicorp = generar_datos_alicorp()

# --- MATPLOTLIB ---
st.header("3. Matplotlib: Tendencias de Crecimiento (Interactivo)")
st.write("Selecciona una o más categorías para ver su volumen de ventas comparativo.")

categorias_disp = list(df_alicorp["Categoria"].unique())
cats_seleccionadas = st.multiselect("Categorías a visualizar:", categorias_disp, default=categorias_disp[:3])

if cats_seleccionadas:
    fig_mpl, ax_mpl = plt.subplots(figsize=(8, 4))
    for cat in cats_seleccionadas:
        resumen_cat = df_alicorp[df_alicorp["Categoria"] == cat].groupby("Marca")["Ventas_Soles"].sum()
        ax_mpl.plot(resumen_cat.index, resumen_cat.values, marker="s", linestyle="--", label=cat)
    
    ax_mpl.set_title("Volumen de Ventas")
    ax_mpl.set_ylabel("Soles")
    ax_mpl.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig_mpl)
else:
    st.warning("Selecciona al menos una categoría.")

st.markdown("---")

# --- SEABORN ---
st.header("4. Seaborn: Análisis de Distribuciones (Interactivo)")
st.write("Explora la distribución de una variable usando un gráfico de cajas.")

variable_y = st.radio("Selecciona la variable a analizar (Eje Y):", ["Ganancias_Soles", "Ventas_Soles", "Costos_Soles"], horizontal=True)

fig_sns, ax_sns = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df_alicorp, x="Marca", y=variable_y, hue="Marca", palette="Set2", legend=False, ax=ax_sns)
ax_sns.set_title(f"Distribución de {variable_y} por Marca")
st.pyplot(fig_sns)
