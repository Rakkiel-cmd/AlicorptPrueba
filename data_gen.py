import pandas as pd
import streamlit as st

@st.cache_data
def cargar_datos_csv():
    """
    Carga los datos directamente desde el archivo físico CSV.
    """
    archivo_csv = "alicorp_simulated_data.csv"
    
    # Obligatoriamente usamos pd.read_csv para leer el archivo del repositorio
    df = pd.read_csv(archivo_csv, parse_dates=["Fecha"])
    return df

if __name__ == "__main__":
    df = cargar_datos_csv()
    print("Datos cargados correctamente desde CSV local.")
