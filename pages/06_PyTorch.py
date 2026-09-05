import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data_gen import cargar_datos_csv

st.set_page_config(page_title="PyTorch", layout="wide", page_icon="🔥")

# TIPOGRAFÍA Y ESTILOS (consistentes con el resto de la app)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"], button, input, textarea {
        font-family: 'Poppins', sans-serif;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
st.sidebar.caption("🔥 Entrenamiento real de un modelo de regresión.")

st.title("PyTorch")
st.write("Entrenamiento de un modelo real de Regresión Lineal usando tensores y redes neuronales simples.")

# PYTORCH
st.header("PyTorch: Predicción de Ganancias")
st.write("Este módulo entrena un modelo lineal de PyTorch para predecir las ganancias en función de las ventas, ajustando los pesos a lo largo de varias épocas para minimizar el Error Cuadrático Medio (MSE).")
st.latex(r"\hat{y} = Wx + b")
st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
st.write("Con Momentum, el modelo acumula \"inercia\" de gradientes anteriores para acelerar la convergencia y reducir oscilaciones:")
st.latex(r"""
\begin{aligned}
v_{t} &= \beta v_{t-1} + \alpha \nabla L(\theta) \\
\theta_{t} &= \theta_{t-1} - v_{t}
\end{aligned}
""")
st.write("Donde $\\beta$ es el **Momentum** y $\\alpha$ es la **Tasa de aprendizaje**.")

df = cargar_datos_csv()
# Datos de entrada (Ventas) y salida (Ganancias)
X_np = df["Ventas_Soles"].values.astype(np.float32).reshape(-1, 1)
y_np = df["Ganancias_Soles"].values.astype(np.float32).reshape(-1, 1)

# Normalizar datos para ayudar a la red a converger más rápido
X_mean, X_std = X_np.mean(), X_np.std()
y_mean, y_std = y_np.mean(), y_np.std()
X_norm = (X_np - X_mean) / X_std
y_norm = (y_np - y_mean) / y_std

# Convertir a tensores de PyTorch
X_tensor = torch.from_numpy(X_norm)
y_tensor = torch.from_numpy(y_norm)

# Modelo de regresión lineal
modelo = nn.Linear(1, 1)

# Función de pérdida y optimizador
criterio = nn.MSELoss()
tasa_aprendizaje = st.slider("Tasa de aprendizaje (Learning Rate):", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
epocas = st.slider("Número de Épocas:", min_value=10, max_value=200, value=50, step=10)
momentum = st.slider("Momentum (Inercia):", min_value=0.0, max_value=0.99, value=0.9, step=0.05)

optimizador = optim.SGD(modelo.parameters(), lr=tasa_aprendizaje, momentum=momentum)

# Botón para iniciar entrenamiento
if st.button("Entrenar Modelo"):
    historial_perdida = []
    
    # Barra de progreso de Streamlit
    barra_progreso = st.progress(0)
    
    for epoca in range(epocas):
        # Forward pass: predecir
        predicciones = modelo(X_tensor)
        perdida = criterio(predicciones, y_tensor)
        
        # Backward pass: optimizar
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        
        historial_perdida.append(perdida.item())
        barra_progreso.progress((epoca + 1) / epocas)
    
    st.success(f"Entrenamiento completado. Pérdida final: {historial_perdida[-1]:.4f}")
    st.balloons()
    
    # Graficar
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disminución del Error")
        with st.container(border=True):
            fig_loss, ax_loss = plt.subplots(figsize=(5,3))
            ax_loss.plot(range(epocas), historial_perdida, color="red")
            ax_loss.set_xlabel("Época")
            ax_loss.set_ylabel("Error (MSE)")
            st.pyplot(fig_loss)
            plt.close(fig_loss)

    with col2:
        st.subheader("Recta de Regresión Aprendida")
        pred_final = modelo(X_tensor).detach().numpy()
        # Des-normalizar para graficar
        pred_real = (pred_final * y_std) + y_mean

        with st.container(border=True):
            fig_reg, ax_reg = plt.subplots(figsize=(5,3))
            ax_reg.scatter(X_np, y_np, alpha=0.3, label="Datos Reales", color="blue")
            ax_reg.plot(X_np, pred_real, color="orange", linewidth=2, label="Línea PyTorch")
            ax_reg.set_xlabel("Ventas (Soles)")
            ax_reg.set_ylabel("Ganancias (Soles)")
            ax_reg.legend()
            st.pyplot(fig_reg)
            plt.close(fig_reg)
