import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_gen import cargar_datos_csv

st.set_page_config(page_title="Matplotlib y Seaborn", layout="wide", page_icon="📉")

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
st.sidebar.caption("📉 Gráficos de tendencia y distribución de datos.")

st.title("Matplotlib y Seaborn")
st.write("Matplotlib arma los gráficos de tendencia y Seaborn analiza cómo se distribuyen los datos.")

df_alicorp = cargar_datos_csv()

tab_mpl, tab_sns = st.tabs(["📈 Matplotlib", "📦 Seaborn"])

# MATPLOTLIB
with tab_mpl:
    st.header("Matplotlib")
    st.write("El gráfico compara las ventas por marca dentro de las categorías seleccionadas.")

    categorias_disp = list(df_alicorp["Categoria"].unique())
    cats_seleccionadas = st.multiselect("Categorías a visualizar:", categorias_disp, default=categorias_disp[:3])

    if cats_seleccionadas:
        with st.container(border=True):
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
with tab_sns:
    st.header("Seaborn")
    st.write("El gráfico de cajas (Boxplot) muestra la distribución, mediana y valores atípicos usando el Rango Intercuartílico (IQR).")
    st.latex(r"IQR = Q_3 - Q_1")

    variable_y = st.radio("Selecciona la variable a analizar (Eje Y):", ["Ganancias_Soles", "Ventas_Soles", "Costos_Soles"], horizontal=True)
    # Comparar por marca o por categoría
    agrupar_por = st.radio("Comparar por:", ["Marca", "Categoria"], horizontal=True)

    with st.container(border=True):
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
