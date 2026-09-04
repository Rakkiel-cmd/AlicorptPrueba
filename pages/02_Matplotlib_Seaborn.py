import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_gen import cargar_o_generar_csv

st.set_page_config(page_title="Matplotlib y Seaborn", layout="wide")

st.title("Matplotlib y Seaborn")
st.write("Matplotlib arma los gráficos de tendencia y Seaborn analiza cómo se distribuyen los datos.")

df_alicorp = cargar_o_generar_csv()

# MATPLOTLIB
st.header("Matplotlib")
st.write("El gráfico compara las ventas por marca dentro de las categorías seleccionadas.")

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
    plt.close(fig_mpl)
else:
    st.warning("Selecciona al menos una categoría.")


# SEABORN
st.header("Seaborn")
st.write("El gráfico de cajas (Boxplot) muestra la distribución, mediana y valores atípicos usando el Rango Intercuartílico (IQR).")
st.latex(r"IQR = Q_3 - Q_1")

variable_y = st.radio("Selecciona la variable a analizar (Eje Y):", ["Ganancias_Soles", "Ventas_Soles", "Costos_Soles"], horizontal=True)
# Comparar por marca o por categoría
agrupar_por = st.radio("Comparar por:", ["Marca", "Categoria"], horizontal=True)

fig_sns, ax_sns = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df_alicorp, x=agrupar_por, y=variable_y, hue=agrupar_por, palette="Set2", legend=False, ax=ax_sns)
ax_sns.set_title(f"Distribución de {variable_y} por {agrupar_por}")
plt.xticks(rotation=30)
st.pyplot(fig_sns)
plt.close(fig_sns)

# Mediana por grupo, debajo del gráfico
st.caption(
    "Mediana por " + agrupar_por + ": " +
    ", ".join(f"{k}: S/ {v:,.0f}" for k, v in df_alicorp.groupby(agrupar_por)[variable_y].median().items())
)
