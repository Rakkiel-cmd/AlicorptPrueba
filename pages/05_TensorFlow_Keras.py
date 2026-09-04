import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from data_gen import cargar_datos_csv

st.set_page_config(page_title="TensorFlow y Keras", layout="wide")

st.title("TensorFlow y Keras")
st.write("TensorFlow es la base para redes neuronales y Keras facilita su creación.")

df_alicorp = cargar_datos_csv()

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
st.write("Este bloque arma la arquitectura de la red. Cada neurona oculta calcula una suma ponderada de sus entradas, sumando un sesgo (bias) y aplicando una función de activación.")
st.latex(r"a^{(l)} = g(W^{(l)}a^{(l-1)} + b^{(l)})")

num_neuronas = st.number_input("Número de Neuronas (Capa Oculta 1):", min_value=8, max_value=128, value=64, step=8)

st.code(
    f'''# Arquitectura del Modelo Secuencial
modelo = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense({num_neuronas}, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])''', language='python'
)

# Construir el modelo real para extraer sus pesos aleatorios
modelo_keras = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense(num_neuronas, activation='relu'),
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

# --- TRAYECTORIA DE OPTIMIZACIÓN Y SUPERFICIE DE PÉRDIDA ---
st.subheader("Superficie de Pérdida y Trayectoria de Entrenamiento (Keras)")
st.write("Entrenamos una pequeña red neuronal (1 neurona, 2 entradas) para predecir si una venta es mayor al promedio. El gráfico muestra cómo desciende el error (loss) a través del espacio de pesos $W_1$ y $W_2$ durante el entrenamiento por gradiente descendente.")

# Preparar datos (Normalizados)
X_keras = df_alicorp[["Edad_Cliente", "Frecuencia_Compra_Mensual"]].astype(float).values
X_keras = (X_keras - X_keras.mean(axis=0)) / X_keras.std(axis=0)
y_keras = (df_alicorp["Ventas_Soles"] > df_alicorp["Ventas_Soles"].median()).astype(int).values

# Modelo Keras
modelo_simple = keras.Sequential([
    keras.layers.Dense(1, use_bias=False, activation='sigmoid', input_shape=(2,))
])
modelo_simple.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5), loss='binary_crossentropy')

# Callback para guardar los pesos en cada época
pesos_historia = []
class GuardaPesos(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pesos_historia.append(self.model.layers[0].get_weights()[0].flatten())

# Iniciar entrenamiento desde un punto aleatorio predefinido para consistencia visual
modelo_simple.layers[0].set_weights([np.array([[-2.0], [2.0]])])
modelo_simple.fit(X_keras, y_keras, epochs=15, verbose=0, callbacks=[GuardaPesos()])
pesos_historia = np.array(pesos_historia)

# Calcular superficie de pérdida
w1_rango = np.linspace(-3.0, 3.0, 30)
w2_rango = np.linspace(-3.0, 3.0, 30)
W1, W2 = np.meshgrid(w1_rango, w2_rango)

def loss_grid(w1, w2, X, y):
    Z = w1 * X[:,0] + w2 * X[:,1]
    A = 1 / (1 + np.exp(-Z))
    A = np.clip(A, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))

Loss = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        Loss[i,j] = loss_grid(W1[i,j], W2[i,j], X_keras, y_keras)

fig_3d = plt.figure(figsize=(10, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')

surf = ax_3d.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.8, edgecolor='none')
loss_historia_val = [loss_grid(w[0], w[1], X_keras, y_keras) for w in pesos_historia]
ax_3d.plot(pesos_historia[:, 0], pesos_historia[:, 1], loss_historia_val, color='red', marker='o', linewidth=2, markersize=5, label='Trayectoria SGD')

ax_3d.set_xlabel('Peso $W_1$ (Edad)')
ax_3d.set_ylabel('Peso $W_2$ (Frecuencia)')
ax_3d.set_zlabel('Pérdida (Cross-Entropy)')
ax_3d.set_title('Superficie de Pérdida en Keras')
ax_3d.legend()

st.pyplot(fig_3d)
plt.close(fig_3d)
