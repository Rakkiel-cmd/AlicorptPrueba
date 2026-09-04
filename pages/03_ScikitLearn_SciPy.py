# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from data_gen import cargar_o_generar_csv

st.set_page_config(page_title="Scikit-Learn y SciPy", layout="wide")

st.title("Scikit-learn y SciPy")
st.write("Scikit-learn agrupa a los clientes con un modelo de Machine Learning, y SciPy analiza la distribución de los tiempos de entrega.")

df_alicorp = cargar_o_generar_csv()

# SCIKIT-LEARN
st.header("Scikit-learn")
st.write("El modelo K-Means agrupa a los clientes minimizando la distancia euclidiana entre los puntos y el centro de su grupo.")
st.latex(r"J = \sum_{j=1}^{K} \sum_{i \in C_j} ||x_i - \mu_j||^2")

num_clusters = st.slider("Número de Clusters (K):", min_value=2, max_value=6, value=3)

X = df_alicorp[["Edad_Cliente", "Frecuencia_Compra_Mensual"]].values
# Edad (18-64) y Frecuencia (1-9) están en escalas muy distintas: sin escalar,
# KMeans agrupaba casi solo por edad. Se normaliza antes de agrupar.
X_esc = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_esc)

fig_skl, ax_skl = plt.subplots(figsize=(8,4))
# Se grafica en escala original (más entendible) aunque el clustering se hizo escalado
scatter = ax_skl.scatter(X[:,0], X[:,1], c=clusters, cmap="viridis", alpha=0.6)
ax_skl.set_xlabel("Edad del Cliente")
ax_skl.set_ylabel("Frecuencia Compra (Mensual)")
st.pyplot(fig_skl)
plt.close(fig_skl)


# SCIPY
st.header("SciPy")
st.write("El gráfico ajusta una curva normal a los tiempos de entrega. La desviación seleccionada define qué tan ancha se ve la curva.")
st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}")

varianza_ajuste = st.slider("Ajuste manual de Desviación (Simulación):", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

tiempos = df_alicorp["Tiempo_Entrega_Dias"].dropna()
mu, std_original = norm.fit(tiempos)
std_simulada = std_original * varianza_ajuste

fig_sci, ax_sci = plt.subplots(figsize=(8, 4))
ax_sci.hist(tiempos, bins=30, density=True, alpha=0.3, color="skyblue", label="Datos Reales (Simulados)")

xmin, xmax = ax_sci.get_xlim()  # antes usaba plt.xlim() sobre estado global; mejor leerlo del propio eje
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std_simulada)

ax_sci.plot(x, p, 'k', linewidth=2, label=rf'Curva Ajustada ($\sigma={std_simulada:.1f}$)')
ax_sci.legend()
ax_sci.set_xlabel("Días de Entrega")
st.pyplot(fig_sci)
plt.close(fig_sci)
