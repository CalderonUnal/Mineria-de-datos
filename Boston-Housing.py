import streamlit as st
import numpy as np
import gzip
import pickle

# Función para cargar el modelo
@st.cache_resource
def load_model():
    filename = "model_trained_regressor.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el modelo entrenado
model = load_model()
# Título de la aplicación
st.title("Predicción de precios de casas - Boston Housing")

st.write("Ingrese las características de la casa para predecir su precio:")

# Crear entradas de texto para las 13 características
features = []
feature_names = [
    "CRIM (Tasa de criminalidad)", "ZN (Proporción de terrenos residenciales)", "INDUS (Proporción de negocios)",
    "CHAS (Variable binaria junto al río)", "NOX (Concentración de óxidos de nitrógeno)", "RM (Número promedio de habitaciones)",
    "AGE (Proporción de casas ocupadas antes de 1940)", "DIS (Distancia a centros de empleo)", "RAD (Índice de accesibilidad a autopistas)",
    "TAX (Tasa de impuesto a la propiedad)", "PTRATIO (Índice alumno-profesor)", "B (Proporción de población negra)", "LSTAT (Porcentaje de población de bajo estatus económico)"
]

for name in feature_names:
    value = st.number_input(f"{name}", value=0.0, step=0.1)
    features.append(value)

# Convertir a array numpy para la predicción
features = np.array(features).reshape(1, -1)

# Botón para predecir
if st.button("Predecir Precio"):
    prediction = model.predict(features)
    st.success(f"El precio estimado de la casa es: ${prediction[0]:,.2f}")

