import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data_gen import generar_datos_alicorp

st.set_page_config(page_title="PyTorch", layout="wide")

st.title("PyTorch")
st.write("Entrenamiento de un modelo real de Regresión Lineal usando tensores y redes neuronales simples.")

# PYTORCH
st.header("PyTorch: Predicción de Ganancias")
st.write("Este módulo entrena un modelo lineal de PyTorch (y = Wx + b) para predecir las ganancias en función de las ventas, ajustando los pesos a lo largo de varias épocas.")

df = generar_datos_alicorp()
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

optimizador = optim.SGD(modelo.parameters(), lr=tasa_aprendizaje)

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
    
    # Graficar
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disminución del Error")
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
        
        fig_reg, ax_reg = plt.subplots(figsize=(5,3))
        ax_reg.scatter(X_np, y_np, alpha=0.3, label="Datos Reales", color="blue")
        ax_reg.plot(X_np, pred_real, color="orange", linewidth=2, label="Línea PyTorch")
        ax_reg.set_xlabel("Ventas (Soles)")
        ax_reg.set_ylabel("Ganancias (Soles)")
        ax_reg.legend()
        st.pyplot(fig_reg)
        plt.close(fig_reg)
