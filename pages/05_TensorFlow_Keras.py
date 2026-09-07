import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from data_gen import cargar_o_generar_csv

st.set_page_config(page_title="TensorFlow y Keras", layout="wide")

st.title("TensorFlow y Keras")
st.write("Predicción de Ganancias basado en Ventas")

df_alicorp = cargar_o_generar_csv()

# ============================================================================
# TENSORFLOW
# ============================================================================
st.header("TensorFlow")

# Datos
X = df_alicorp["Ventas_Soles"].values.astype("float32").reshape(-1, 1)
y = df_alicorp["Ganancias_Soles"].values.astype("float32").reshape(-1, 1)

# Normalizar
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# Crear modelo simple
modelo_tf = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])

modelo_tf.compile(optimizer='adam', loss='mse')

# Entrenar
st.write("Entrenando modelo...")
historial = modelo_tf.fit(X_norm, y_norm, epochs=50, verbose=0)

st.success("✅ Entrenamiento completado")

# Predicciones
predicciones_norm = modelo_tf.predict(X_norm, verbose=0)
predicciones = (predicciones_norm * y_std) + y_mean

# Tabla de predicciones
st.subheader("📊 Predicciones - TensorFlow")

df_pred = pd.DataFrame({
    "Ventas": X.flatten()[:15],
    "Ganancias_Reales": y.flatten()[:15],
    "Ganancias_Predichas": np.round(predicciones.flatten()[:15], 2),
    "Error": np.round(np.abs(y.flatten()[:15] - predicciones.flatten()[:15]), 2)
})

st.dataframe(df_pred, use_container_width=True)

# Gráfico simple
fig, ax = plt.subplots()
ax.scatter(X, y, alpha=0.5, label="Real")
ax.scatter(X, predicciones, alpha=0.5, color="red", label="Predicción")
ax.set_xlabel("Ventas")
ax.set_ylabel("Ganancias")
ax.legend()
st.pyplot(fig)

# ============================================================================
# KERAS
# ============================================================================
st.header("Keras")

# Datos binarios (Alto o Bajo)
X_keras = df_alicorp[["Edad_Cliente", "Frecuencia_Compra_Mensual"]].values.astype("float32")
X_keras = (X_keras - X_keras.mean(axis=0)) / X_keras.std(axis=0)
y_keras = (df_alicorp["Ventas_Soles"] > df_alicorp["Ventas_Soles"].median()).astype(int).values

# Crear modelo
modelo_keras = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

modelo_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar
st.write("Entrenando modelo...")
modelo_keras.fit(X_keras, y_keras, epochs=50, verbose=0)

st.success("✅ Entrenamiento completado")

# Predicciones
predicciones_keras = modelo_keras.predict(X_keras, verbose=0)

# Tabla de predicciones
st.subheader("📊 Predicciones - Keras")

df_pred_keras = pd.DataFrame({
    "Edad": X_keras[:15, 0],
    "Frecuencia": X_keras[:15, 1],
    "Probabilidad": np.round(predicciones_keras.flatten()[:15], 3),
    "Predicción": ["Alto" if p > 0.5 else "Bajo" for p in predicciones_keras.flatten()[:15]],
    "Real": ["Alto" if val == 1 else "Bajo" for val in y_keras[:15]]
})

st.dataframe(df_pred_keras, use_container_width=True)

# Gráfico simple
fig2, ax2 = plt.subplots()
ax2.hist(predicciones_keras, bins=20, alpha=0.7)
ax2.set_xlabel("Probabilidad")
ax2.set_ylabel("Frecuencia")
ax2.set_title("Distribución de Predicciones")
st.pyplot(fig2)