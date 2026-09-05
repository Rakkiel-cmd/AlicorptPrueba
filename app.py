import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_gen import cargar_datos_csv

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Alicorp Analytics & ML", layout="wide", page_icon="📈")

# Generar y cargar los datos (Ficticios)
df_alicorp = cargar_datos_csv()

# TIPOGRAFÍA Y ESTILOS GLOBALES (look & feel de página web)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"], button, input, textarea {
        font-family: 'Poppins', sans-serif;
    }
    div[data-testid="stMetric"] {
        background-color: #FFF6F2;
        border: 1px solid #F0C9BA;
        border-radius: 12px;
        padding: 16px 14px 12px 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        color: #E4572E !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# LOGO ORIGINAL (ícono propio, no el logo real de Alicorp) + MENÚ LATERAL
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
st.sidebar.markdown("Proyecto de Machine Learning - Alicorp")
st.sidebar.caption("📊 Dashboard: KPIs y evolución de ventas.")

# BANNER DE ENCABEZADO (con logo original en SVG, no la marca real de Alicorp)
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #E4572E 0%, #C7431F 100%); padding: 22px 28px;
                border-radius: 14px; margin-bottom: 20px; display:flex; align-items:center; gap:16px;
                box-shadow: 0 4px 14px rgba(228,87,46,0.25);">
        <div style="background-color:rgba(255,255,255,0.18); border-radius:10px; padding:10px; display:flex;">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="12" width="4" height="9" rx="1" fill="white"/>
                <rect x="10" y="7" width="4" height="14" rx="1" fill="white"/>
                <rect x="17" y="3" width="4" height="18" rx="1" fill="white"/>
            </svg>
        </div>
        <div>
            <h2 style="color:white; margin:0; font-weight:600;">Alicorp Analytics &amp; ML</h2>
            <p style="color:#FCE9E2; margin:4px 0 0 0; font-size:15px;">
                Panel interactivo de analítica comercial y Machine Learning — Proyecto SENATI
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# DASHBOARD 1: ANÁLISIS COMERCIAL (HOME)
st.title("Dashboard Alicorp")
st.write("Análisis principal que muestra las ventas, las ganancias y el tiempo de entrega promedio de los productos.")

st.caption("Los datos son simulados por el código, no corresponden a ventas reales de Alicorp.")

# KPIs
st.subheader("Indicadores Clave de Rendimiento (KPIs)")
st.latex(r"\Delta \% = \left( \frac{\text{Valor}_{actual} - \text{Valor}_{anterior}}{\text{Valor}_{anterior}} \right) \times 100")
col1, col2, col3, col4 = st.columns(4)

ventas_totales = df_alicorp["Ventas_Soles"].sum()
ganancias_totales = df_alicorp["Ganancias_Soles"].sum()
promedio_entrega = df_alicorp["Tiempo_Entrega_Dias"].mean()
promedio_venta = df_alicorp["Ventas_Soles"].mean()
mediana_venta = df_alicorp["Ventas_Soles"].median()

# Cálculos reales de deltas: ventana móvil de 30 días vs los 30 días previos a esos
# (antes se comparaba mes calendario contra mes calendario, lo cual daba variaciones
# falsas cuando el "mes actual" tenía muchos menos días de datos que el anterior)
fecha_max = df_alicorp["Fecha"].max()
inicio_actual = fecha_max - pd.Timedelta(days=30)
inicio_anterior = fecha_max - pd.Timedelta(days=60)

df_periodo_actual = df_alicorp[df_alicorp["Fecha"] > inicio_actual]
df_periodo_anterior = df_alicorp[(df_alicorp["Fecha"] > inicio_anterior) & (df_alicorp["Fecha"] <= inicio_actual)]

# Evitar división por cero
ventas_ant = df_periodo_anterior["Ventas_Soles"].sum() if not df_periodo_anterior.empty else 1
ganancias_ant = df_periodo_anterior["Ganancias_Soles"].sum() if not df_periodo_anterior.empty else 1
entrega_ant = df_periodo_anterior["Tiempo_Entrega_Dias"].mean() if not df_periodo_anterior.empty else promedio_entrega

delta_ventas = ((df_periodo_actual["Ventas_Soles"].sum() - ventas_ant) / ventas_ant) * 100
delta_ganancias = ((df_periodo_actual["Ganancias_Soles"].sum() - ganancias_ant) / ganancias_ant) * 100
delta_entrega = df_periodo_actual["Tiempo_Entrega_Dias"].mean() - entrega_ant

col1.metric("Ventas Totales (Simuladas)", f"S/ {ventas_totales:,.2f}", f"{delta_ventas:+.1f}% vs 30 días previos")
col2.metric("Ganancias Netas (Simuladas)", f"S/ {ganancias_totales:,.2f}", f"{delta_ganancias:+.1f}% vs 30 días previos")
col3.metric("Venta Promedio / Mediana", f"S/ {promedio_venta:,.0f} / S/ {mediana_venta:,.0f}")
col4.metric("Tiempo Entrega Promedio", f"{promedio_entrega:.1f} días", f"{delta_entrega:+.1f} días vs 30 días previos", delta_color="inverse")

# Serie de Tiempo (Ventas a lo largo del tiempo)
st.subheader("Evolución de Ventas a lo Largo del Tiempo")
df_series = df_alicorp.groupby(df_alicorp["Fecha"].dt.to_period("M"))["Ventas_Soles"].sum().reset_index()
df_series["Fecha"] = df_series["Fecha"].dt.to_timestamp()

with st.container(border=True):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df_series, x="Fecha", y="Ventas_Soles", marker="o", color="blue", ax=ax)
    ax.set_title("Ventas Mensuales Históricas (Soles)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Ventas Totales")
    st.pyplot(fig)
    plt.close(fig)  # evita acumular figuras en memoria cada vez que Streamlit vuelve a correr el script

# --- TABLA DE DATOS ---
st.markdown("---")
st.subheader("🗃️ Base de Datos de Ventas de Alicorp")
st.write("Registros de clientes, ventas y logística recopilados.")

# Mostrar el DataFrame completo con estilo interactivo
st.dataframe(df_alicorp)

# --- PIE DE PÁGINA ---
st.markdown(
    """
    <hr style="margin-top:36px; margin-bottom:14px; border:none; border-top:1px solid #F0C9BA;">
    <div style="text-align:center; color:#8a8a8a; font-size:13px; padding-bottom:8px;">
        Proyecto SENATI 2026 · Alicorp Analytics &amp; ML<br>
        Desarrollado por <strong>Rakkiel-cmd</strong>, <strong>LeonardoLatorreH</strong> y <strong>Dayra1903</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
