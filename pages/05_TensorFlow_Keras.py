import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from data_gen import cargar_datos_csv

st.set_page_config(page_title="TensorFlow y Keras", layout="wide", page_icon="🧠")

# TIPOGRAFÍA Y ESTILOS (consistentes con el resto de la app)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"], button, input, textarea {
        font-family: 'Poppins', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FDF6F2 100%);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    @keyframes logoGlowPulse {
        0% { box-shadow: 0 0 0 0 rgba(228,87,46,0.55); }
        70% { box-shadow: 0 0 0 14px rgba(228,87,46,0); }
        100% { box-shadow: 0 0 0 0 rgba(228,87,46,0); }
    }
    .logo-glow {
        animation: logoGlowPulse 2.2s infinite;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
        <div class="logo-glow" style="background:#E4572E; border-radius:8px; padding:6px; display:flex;">
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
st.sidebar.caption("🧠 Arquitectura y entrenamiento de redes neuronales.")

st.title("TensorFlow y Keras")
st.write("TensorFlow es la base para redes neuronales y Keras facilita su creación.")

df_alicorp = cargar_datos_csv()

tab_tf, tab_keras = st.tabs(["🧮 TensorFlow", "🧠 Keras"])

# TENSORFLOW
with tab_tf:
    st.header("TensorFlow")
    st.write("Esta red neuronal se construye y entrena con operaciones de bajo nivel de TensorFlow (tf.Variable y tf.GradientTape), sin usar la API de alto nivel de Keras. Así se ve, paso a paso, lo que Keras hace \"por dentro\" al entrenar un modelo.")
    st.latex(r"\hat{y} = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2")

    tasa_aprendizaje_tf = st.slider("Tasa de Aprendizaje (Learning Rate):", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
    epocas_tf = st.slider("Número de Épocas:", min_value=10, max_value=100, value=40, step=10)

    # Datos reales: predecir Ganancias en función de Ventas (normalizados para ayudar a converger)
    X_tf_np = df_alicorp["Ventas_Soles"].values.astype("float32").reshape(-1, 1)
    y_tf_np = df_alicorp["Ganancias_Soles"].values.astype("float32").reshape(-1, 1)
    X_mean_tf, X_std_tf = X_tf_np.mean(), X_tf_np.std()
    y_mean_tf, y_std_tf = y_tf_np.mean(), y_tf_np.std()
    X_tensor_tf = tf.constant((X_tf_np - X_mean_tf) / X_std_tf)
    y_tensor_tf = tf.constant((y_tf_np - y_mean_tf) / y_std_tf)

    # Pesos de una red neuronal de 2 capas (1 entrada -> 8 neuronas ocultas -> 1 salida),
    # creados y entrenados manualmente con TensorFlow (sin usar keras.Sequential)
    tf.random.set_seed(42)
    w1_tf = tf.Variable(tf.random.normal([1, 8], stddev=0.5))
    b1_tf = tf.Variable(tf.zeros([8]))
    w2_tf = tf.Variable(tf.random.normal([8, 1], stddev=0.5))
    b2_tf = tf.Variable(tf.zeros([1]))

    historial_perdida_tf = []
    barra_tf = st.progress(0)
    for epoca in range(epocas_tf):
        with tf.GradientTape() as tape:
            capa_oculta = tf.nn.relu(tf.matmul(X_tensor_tf, w1_tf) + b1_tf)
            prediccion_tf = tf.matmul(capa_oculta, w2_tf) + b2_tf
            perdida_tf = tf.reduce_mean(tf.square(prediccion_tf - y_tensor_tf))

        gradientes_tf = tape.gradient(perdida_tf, [w1_tf, b1_tf, w2_tf, b2_tf])
        w1_tf.assign_sub(tasa_aprendizaje_tf * gradientes_tf[0])
        b1_tf.assign_sub(tasa_aprendizaje_tf * gradientes_tf[1])
        w2_tf.assign_sub(tasa_aprendizaje_tf * gradientes_tf[2])
        b2_tf.assign_sub(tasa_aprendizaje_tf * gradientes_tf[3])

        historial_perdida_tf.append(float(perdida_tf))
        barra_tf.progress((epoca + 1) / epocas_tf)

    with st.container(border=True):
        fig_tf, ax_tf = plt.subplots(figsize=(8,4))
        ax_tf.plot(range(epocas_tf), historial_perdida_tf, color="#E4572E", linewidth=2)
        ax_tf.set_xlabel("Época")
        ax_tf.set_ylabel("Error (MSE)")
        ax_tf.set_title("Disminución del Error - Red Neuronal en TensorFlow")
        st.pyplot(fig_tf)
        plt.close(fig_tf)

    st.caption(f"Pérdida final tras {epocas_tf} épocas: {historial_perdida_tf[-1]:.4f}")


# KERAS
with tab_keras:
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
    with st.container(border=True):
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
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(1, use_bias=False, activation='sigmoid')
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

    with st.container(border=True):
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
