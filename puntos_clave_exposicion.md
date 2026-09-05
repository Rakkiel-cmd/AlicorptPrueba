# Puntos clave para la exposición — Redes Neuronales

Guía rápida para el delegado/expositor. El instructor pidió que las redes neuronales estén **explicadas a detalle**, no solo mostradas funcionando. Aquí está el "por qué" detrás de cada decisión.

---

## 1. Página Keras / TensorFlow (`pages/05_TensorFlow_Keras.py`)

**¿Qué arquitectura usamos y por qué?**
```
Input(shape=(10,)) → Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)
```
- Es una red **secuencial** (cada capa alimenta a la siguiente, sin ramificaciones).
- Las capas van reduciendo neuronas (64 → 32 → 1) — patrón típico de "embudo": las primeras capas capturan combinaciones amplias de las variables de entrada, las últimas van sintetizando hacia una sola salida.
- `relu` en las capas ocultas: activa solo las neuronas con valores positivos, ayuda a que la red aprenda relaciones no lineales sin "apagarse" (a diferencia de sigmoid, que sí se satura en las capas intermedias).
- `sigmoid` en la capa final: comprime la salida entre 0 y 1 — apropiado porque es un problema de clasificación binaria (por eso se usa también en el modelo simplificado de 1 neurona).

**¿Qué muestra el heatmap de pesos?**
Los pesos de la primera capa (64 neuronas) **antes de entrenar** — son valores aleatorios iniciales. Sirve para mostrar visualmente que una red arranca sin ningún conocimiento; todo lo que "aprende" viene después, ajustando estos pesos con los datos.

**¿Qué es la superficie de pérdida 3D?**
Entrenamos una red simplificada de **1 neurona con 2 entradas** (edad del cliente y frecuencia de compra) para predecir si una venta supera el promedio. Como solo hay 2 pesos ($W_1$, $W_2$), se puede graficar el error (pérdida) para *todas* las combinaciones posibles de esos 2 pesos — esa es la "superficie". La línea roja es la trayectoria real que siguió el optimizador (SGD) bajando hacia el punto más bajo (el error mínimo) durante el entrenamiento. Es una forma visual de explicar qué significa "entrenar": ir bajando por esa superficie hasta el mínimo.

---

## 2. Página PyTorch (`pages/06_PyTorch.py`)

**¿Qué hace este modelo?**
Es una regresión lineal real ($\hat{y} = Wx + b$): predice las **ganancias** de una venta a partir de las **ventas totales**. Se entrena de verdad (no es una simulación visual como en TensorFlow) — cada vez que se presiona "Entrenar Modelo", el peso $W$ y el sesgo $b$ se ajustan con datos reales del CSV.

**¿Qué es el MSE y por qué se usa?**
Error Cuadrático Medio: el promedio de (predicción − valor real)² para todos los datos. Se eleva al cuadrado para que los errores grandes pesen más y para que no se cancelen los positivos con los negativos. Es la métrica estándar para problemas de regresión (predecir un número, no una categoría).

**¿Qué es el Momentum y por qué se agregó?**
```
v_t = β·v_{t-1} + α·∇L(θ)
θ_t = θ_{t-1} - v_t
```
Sin Momentum, el descenso de gradiente (SGD) puede avanzar muy lento o "zigzaguear" cuando el terreno del error es irregular. Momentum ($\beta$, el slider "Inercia") hace que el optimizador **acumule velocidad** en la dirección en la que ya venía bajando, igual que una pelota rodando cuesta abajo — converge más rápido y con menos oscilaciones. $\alpha$ es la tasa de aprendizaje (qué tan grande es cada paso).

**Qué mostrar en vivo durante la exposición:**
1. Mover el slider de Momentum a 0 y entrenar → señalar que baja más lento/con más ruido.
2. Subir Momentum a 0.9 y entrenar de nuevo → señalar que baja más rápido y liso.
3. Mostrar la gráfica de "Recta de Regresión Aprendida" y explicar que la línea naranja es literalmente lo que el modelo aprendió a partir de los puntos azules (datos reales).

---

## 3. Posibles preguntas del instructor (y respuesta corta)

- **"¿Por qué Random Forest y no otro modelo?"** (página Scikit-learn) → Porque es un problema de clasificación con varias variables numéricas y categorías, y Random Forest maneja bien relaciones no lineales sin necesitar normalizar los datos primero.
- **"¿Los datos son reales?"** → No, son simulados con código (`alicorp_simulated_data.csv`), se aclara explícitamente en el dashboard. El objetivo es demostrar las librerías, no hacer un análisis real de Alicorp.
- **"¿Qué diferencia hay entre la red de Keras y la de PyTorch?"** → La de Keras (superficie 3D) es para *visualizar* cómo se ve un entrenamiento por dentro; la de PyTorch es un modelo real y interactivo que el usuario entrena en el momento con sus propios parámetros (learning rate, épocas, momentum).
