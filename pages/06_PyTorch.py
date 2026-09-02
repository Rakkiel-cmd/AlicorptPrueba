import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch

st.set_page_config(page_title="PyTorch", layout="wide")

st.title("PyTorch")
st.write("PyTorch trabaja con tensores, la estructura de datos que usan las redes neuronales para procesar la información.")

# PYTORCH
st.header("PyTorch")
st.write("El heatmap muestra la correlación entre las características del tensor. La dimensión seleccionada define su tamaño.")

dimension_tensor = st.slider("Dimensión del Tensor (Características):", min_value=3, max_value=12, value=5)

# Crear un tensor de características aleatorio usando PyTorch de tamaño dinámico
tensor_caracteristicas = torch.rand(dimension_tensor, dimension_tensor)

# Hacerlo simétrico como una matriz de correlación
tensor_correlacion = (tensor_caracteristicas + tensor_caracteristicas.T) / 2
tensor_correlacion.fill_diagonal_(1.0)

fig_torch, ax_torch = plt.subplots(figsize=(6, 5))
sns.heatmap(tensor_correlacion.numpy(), annot=True, cmap="coolwarm", ax=ax_torch, fmt=".2f")
ax_torch.set_title(f"Heatmap de Correlación de Tensor PyTorch ({dimension_tensor}x{dimension_tensor})")
st.pyplot(fig_torch)
plt.close(fig_torch)
