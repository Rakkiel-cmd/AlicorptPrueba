import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_gen import generar_datos_alicorp

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Alicorp Analytics & ML", layout="wide", page_icon="📈")

# Generar y cargar los datos (Ficticios)
df_alicorp = generar_datos_alicorp()

# MENÚ LATERAL
st.sidebar.markdown("### Alicorp Analytics")
st.sidebar.markdown("Proyecto de Machine Learning - Alicorp")

# DASHBOARD 1: ANÁLISIS COMERCIAL (HOME)
st.title("Dashboard Alicorp")
st.write("El panel principal reúne las ventas, las ganancias y el tiempo de entrega del período. Cada librería se desarrolla en una página distinta del menú.")

st.caption("Los datos son generados por el código de forma aleatoria, no corresponden a ventas reales de Alicorp.")

# KPIs
st.subheader("Indicadores Clave de Rendimiento (KPIs)")
col1, col2, col3, col4 = st.columns(4)

ventas_totales = df_alicorp["Ventas_Soles"].sum()
ganancias_totales = df_alicorp["Ganancias_Soles"].sum()
promedio_entrega = df_alicorp["Tiempo_Entrega_Dias"].mean()
# Promedio y mediana de venta por pedido (antes solo se mostraba el total)
promedio_venta = df_alicorp["Ventas_Soles"].mean()
mediana_venta = df_alicorp["Ventas_Soles"].median()

col1.metric("Ventas Totales (Simuladas)", f"S/ {ventas_totales:,.2f}", "+5.2% vs mes anterior")
col2.metric("Ganancias Netas (Simuladas)", f"S/ {ganancias_totales:,.2f}", "+2.1% vs mes anterior")
col3.metric("Venta Promedio / Mediana", f"S/ {promedio_venta:,.0f} / S/ {mediana_venta:,.0f}")
col4.metric("Tiempo Entrega Promedio", f"{promedio_entrega:.1f} días", "-0.5 días vs mes anterior", delta_color="inverse")

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
plt.close(fig)  # evita acumular figuras en memoria cada vez que Streamlit vuelve a correr el script
