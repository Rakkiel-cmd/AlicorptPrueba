import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_gen import generar_datos_alicorp

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Alicorp Analytics & ML", layout="wide", page_icon="📈")

# Generar y cargar los datos (Ficticios)
df_alicorp = generar_datos_alicorp()

# --- MENÚ LATERAL ---
st.sidebar.markdown("### 🛒 Alicorp Analytics")
st.sidebar.markdown("---")
st.sidebar.markdown("**Desarrollado para:** Proyecto Analítico Alicorp")
st.sidebar.markdown("**Nota:** Los datos son generados aleatoriamente.")

# --- DASHBOARD 1: ANÁLISIS COMERCIAL (HOME) ---
st.title("📈 Dashboard Principal: Análisis Comercial Alicorp")
st.write("Bienvenido al sistema analítico demostrativo. A la izquierda tienes el menú para navegar por las librerías.")

st.info("💡 **Aviso para la Evaluación:** Todos los datos monetarios, fechas y métricas mostrados aquí han sido generados de manera simulada/ficticia en el código para propósitos académicos y no corresponden a datos reales de Alicorp.")

# KPIs
st.subheader("Indicadores Clave de Rendimiento (KPIs)")
col1, col2, col3 = st.columns(3)

ventas_totales = df_alicorp["Ventas_Soles"].sum()
ganancias_totales = df_alicorp["Ganancias_Soles"].sum()
promedio_entrega = df_alicorp["Tiempo_Entrega_Dias"].mean()

col1.metric("Ventas Totales (Simuladas)", f"S/ {ventas_totales:,.2f}", "+5.2% vs mes anterior")
col2.metric("Ganancias Netas (Simuladas)", f"S/ {ganancias_totales:,.2f}", "+2.1% vs mes anterior")
col3.metric("Tiempo Entrega Promedio", f"{promedio_entrega:.1f} días", "-0.5 días vs mes anterior", delta_color="inverse")

st.markdown("---")

# Serie de Tiempo (Ventas a lo largo del tiempo)
st.subheader("Evolución de Ventas a lo Largo del Tiempo")
df_series = df_alicorp.groupby(df_alicorp["Fecha"].dt.to_period("M"))["Ventas_Soles"].sum().reset_index()
df_series["Fecha"] = df_series["Fecha"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_series, x="Fecha", y="Ventas_Soles", marker="o", color="blue", ax=ax)
ax.set_title("Ventas Mensuales Históricas (Soles)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Ventas Totales")
st.pyplot(fig)
