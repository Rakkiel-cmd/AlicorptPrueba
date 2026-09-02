import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="TensorFlow y Keras", layout="wide")

st.title("TensorFlow y Keras")
st.write("Ambas librerías se usan para construir y entrenar redes neuronales, en este caso orientadas a predecir el abandono de clientes.")

# TENSORFLOW
st.header("TensorFlow")
st.write("La curva muestra cómo baja el error del modelo en cada época. La tasa de aprendizaje cambia qué tan rápido baja.")

tasa_aprendizaje = st.slider("Tasa de Aprendizaje (Simulada):", min_value=1.0, max_value=10.0, value=5.0, step=1.0)

# Curva simulada (no es un entrenamiento real) solo para mostrar el efecto visual de la tasa de aprendizaje
epocas = np.arange(1, 21)
loss = np.exp(-epocas/tasa_aprendizaje) + np.random.normal(0, 0.05, 20)
val_loss = np.exp(-epocas/(tasa_aprendizaje*0.9)) + np.random.normal(0, 0.08, 20)

fig_tf, ax_tf = plt.subplots(figsize=(8,4))
ax_tf.plot(epocas, loss, label="Pérdida Entrenamiento (Loss)")
ax_tf.plot(epocas, val_loss, label="Pérdida Validación (Val Loss)")
ax_tf.set_xlabel("Épocas")
ax_tf.set_ylabel("Pérdida")
ax_tf.legend()
st.pyplot(fig_tf)
plt.close(fig_tf)


# KERAS
st.header("Keras")
st.write("Este bloque arma la arquitectura de la red y muestra los pesos iniciales de la capa oculta según el número de neuronas elegido.")

num_neuronas = st.number_input("Número de Neuronas (Capa Oculta 1):", min_value=8, max_value=128, value=64, step=8)

st.code(
    f'''# Arquitectura del Modelo Secuencial
modelo = keras.Sequential([
    keras.layers.Dense({num_neuronas}, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])''', language='python'
)

# Construir el modelo real para extraer sus pesos aleatorios
modelo_keras = keras.Sequential([
    keras.layers.Dense(num_neuronas, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

st.write(f"El heatmap siguiente muestra los pesos iniciales de las {num_neuronas} neuronas, antes de cualquier entrenamiento.")
pesos = modelo_keras.layers[0].get_weights()[0]
fig_keras, ax_keras = plt.subplots(figsize=(10,2))
sns.heatmap(pesos.T, cmap="YlGnBu", cbar=False, ax=ax_keras)
ax_keras.set_title(f"Heatmap de Pesos Iniciales (Capa Densa 1 con {num_neuronas} neuronas)")
st.pyplot(fig_keras)
plt.close(fig_keras)
