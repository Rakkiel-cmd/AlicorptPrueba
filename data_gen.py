import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import random
from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
import streamlit as st

@st.cache_data
def generar_datos_alicorp(n_registros=1000):
    '''
    ATENCIÓN PROFESOR/EVALUADOR: 
    Todos los datos generados por esta función son 100% FICTICIOS y SIMULADOS.
    Han sido creados exclusivamente con fines académicos para demostrar el 
    funcionamiento de las librerías solicitadas en el proyecto.
    No representan cifras reales de la empresa Alicorp.
    '''
    # Semilla fija: sin esto, cada refresco de la app generaba números distintos
    # y el CSV de ejemplo del repo nunca coincidía con lo mostrado en vivo.
    random.seed(42)
    np.random.seed(42)

    marcas = ["Primor", "Blanca Flor", "Bolívar", "Don Vittorio", "Casino", "Sayón"]
    categorias = ["Aceites", "Harinas", "Detergentes", "Fideos", "Galletas", "Chocolates"]
    
    # Fechas simuladas del último año
    fechas = [datetime.today() - timedelta(days=random.randint(0, 365)) for _ in range(n_registros)]
    
    # Valores monetarios simulados.
    # Con normal(150, 50) algunas ventas caían en negativo (imposible en la realidad
    # y arrastraba costos/ganancias negativos también). Se acota con un piso mínimo.
    ventas_s = np.random.normal(loc=150, scale=50, size=n_registros)
    ventas_s = np.clip(ventas_s, 10, None)
    costos_s = ventas_s * np.random.uniform(0.4, 0.7, size=n_registros)
    ganancias_s = ventas_s - costos_s
    
    # Tiempos de logística
    tiempos_entrega = np.random.lognormal(mean=1.5, sigma=0.5, size=n_registros)
    
    # Reseñas falsas
    reseñas = [
        "Muy buen producto, siempre compro de esta marca.",
        "Pésima calidad, vino abierto el empaque.",
        "Cumple su función, precio razonable.",
        "Me encanta Blanca Flor para mis postres.",
        "El aceite Primor rinde muchísimo.",
        "El detergente no quitó las manchas.",
        "Excelente sabor de las galletas.",
        "Muy caro para lo que ofrece."
    ]
    
    data = {
        "Fecha": fechas,
        "Marca": np.random.choice(marcas, n_registros),
        "Categoria": np.random.choice(categorias, n_registros),
        "Ventas_Soles": ventas_s,
        "Costos_Soles": costos_s,
        "Ganancias_Soles": ganancias_s,
        "Tiempo_Entrega_Dias": tiempos_entrega,
        "Reseña_Cliente": np.random.choice(reseñas, n_registros),
        "Edad_Cliente": np.random.randint(18, 65, size=n_registros),
        "Frecuencia_Compra_Mensual": np.random.randint(1, 10, size=n_registros)
    }
    
    df = pd.DataFrame(data)
    
    # Guardar localmente como CSV para acceso rápido / evidencia de "consumo de CSV"
    df.to_csv("alicorp_simulated_data.csv", index=False)
    
    return df
