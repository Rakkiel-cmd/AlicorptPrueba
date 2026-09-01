import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import norm
from data_gen import generar_datos_alicorp

st.set_page_config(page_title="Scikit-Learn y SciPy", layout="wide")

st.title("🧩 Integrante 3: Scikit-learn y SciPy")
st.write("Modelos de Machine Learning Tradicional y Distribuciones Estadísticas.")

df_alicorp = generar_datos_alicorp()

# --- SCIKIT-LEARN ---
st.header("5. Scikit-learn: Segmentación de Clientes (Interactivo)")
st.write("Ajusta el algoritmo K-Means alterando el número de clusters deseados para segmentar a los clientes.")

num_clusters = st.slider("Número de Clusters (K):", min_value=2, max_value=6, value=3)

X = df_alicorp[["Edad_Cliente", "Frecuencia_Compra_Mensual"]].values
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

fig_skl, ax_skl = plt.subplots(figsize=(8,4))
scatter = ax_skl.scatter(X[:,0], X[:,1], c=clusters, cmap="viridis", alpha=0.6)
ax_skl.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c="red", marker="X", label="Centroides")
ax_skl.set_xlabel("Edad del Cliente")
ax_skl.set_ylabel("Frecuencia Compra (Mensual)")
ax_skl.legend()
st.pyplot(fig_skl)

st.markdown("---")

# --- SCIPY ---
st.header("6. SciPy: Análisis Logístico de Tiempos de Entrega (Interactivo)")
st.write("Visualiza cómo la variabilidad (desviación estándar) afecta la campana de Gauss de los tiempos de entrega.")

varianza_ajuste = st.slider("Ajuste manual de Desviación (Simulación):", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

tiempos = df_alicorp["Tiempo_Entrega_Dias"].dropna()
mu, std_original = norm.fit(tiempos)
std_simulada = std_original * varianza_ajuste

fig_sci, ax_sci = plt.subplots(figsize=(8, 4))
ax_sci.hist(tiempos, bins=30, density=True, alpha=0.3, color="skyblue", label="Datos Reales (Simulados)")

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std_simulada)

ax_sci.plot(x, p, 'k', linewidth=2, label=rf'Curva Ajustada ($\sigma={std_simulada:.1f}$)')
ax_sci.legend()
ax_sci.set_xlabel("Días de Entrega")
st.pyplot(fig_sci)
