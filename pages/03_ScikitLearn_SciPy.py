# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import norm
from data_gen import cargar_datos_csv

st.set_page_config(page_title="Scikit-Learn y SciPy", layout="wide")

st.title("Scikit-learn y SciPy")
st.write("Scikit-learn agrupa a los clientes con un modelo de Machine Learning, y SciPy analiza la distribución de los tiempos de entrega.")

df_alicorp = cargar_datos_csv()

# SCIKIT-LEARN
st.header("Scikit-learn")
st.write("El modelo Random Forest clasifica la Categoría de producto basándose en características de los clientes. A continuación se muestra la Matriz de Confusión del modelo.")
st.latex(r"C_{i,j} = \sum_{k=1}^{N} \mathbb{I}(y_k = i \land \hat{y}_k = j)")

# Preparar datos
X = df_alicorp[["Edad_Cliente", "Ventas_Soles", "Tiempo_Entrega_Dias"]].astype(float)
y = df_alicorp["Categoria"].astype(str)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

fig_skl, ax_skl = plt.subplots(figsize=(8, 5))
disp.plot(ax=ax_skl, cmap="Blues", xticks_rotation=45)
ax_skl.set_title("Matriz de Confusión - Clasificación de Categoría")
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
