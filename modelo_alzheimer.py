import streamlit as st
import numpy as np
import gzip
import pickle

# Función para cargar el modelo
@st.cache_resource
def load_model():
    filename = "modelo_alzheimer.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el modelo entrenado
model = load_model()

# Título de la aplicación
st.title("Predicción Alzheimer")

# Mostrar información sobre el modelo seleccionado
st.subheader("Modelo Seleccionado")
st.write("Modelo de clasificación de Alzheimer basado en características clínicas.")

# Definir las características del modelo
features = [
    "Edad", "Género", "Educación", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF",
    "Hipocampo Izquierdo", "Hipocampo Derecho", "Tercer Ventrículo", "Cuarto Ventrículo",
    "Cisterna Magna", "Volumen de la sustancia blanca", "Grosor cortical promedio",
    "Volumen del giro cingulado anterior", "Índice de atrofia ventricular",
    "Volumen del tálamo izquierdo", "Volumen del tálamo derecho", "Volumen del caudado izquierdo",
    "Volumen del caudado derecho", "Volumen del putamen izquierdo", "Volumen del putamen derecho"
]

# Crear entradas para cada característica
user_input = []
for feature in features:
    value = st.number_input(f"{feature}", value=0.0, format="%.2f")
    user_input.append(value)

# Botón para hacer la predicción
if st.button("Predecir"): 
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    resultado = "Positivo para Alzheimer" if prediction[0] == 1 else "Negativo para Alzheimer"
    st.subheader("Resultado de la Predicción")
    st.write(resultado)
