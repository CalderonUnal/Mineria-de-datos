import streamlit as st
import numpy as np
import gzip
import pickle
import pandas as pd

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

# Definir las características del dataset
numeric_features = ['Age', 'Education Level', 'Cognitive Test Score']
continuous_features = ['BMI']

categorical_features = {
    'Gender': ['Male', 'Female'],
    'Physical Activity Level': ['Low', 'Medium', 'High'],
    'Smoking Status': ['Never', 'Former', 'Current'],
    'Alcohol Consumption': ['Never', 'Occasionally', 'Regularly'],
    'Diabetes': ['No', 'Yes'],
    'Hypertension': ['No', 'Yes'],
    'Family History of Alzheimer’s': ['No', 'Yes'],
    'Dietary Habits': ['Unhealthy', 'Average', 'Healthy'],
    'Employment Status': ['Unemployed', 'Employed', 'Retired'],
    'Marital Status': ['Single', 'Married', 'Widowed'],
    'Genetic Risk Factor (APOE-ε4 allele)': ['No', 'Yes'],
    'Social Engagement Level': ['Low', 'Medium', 'High'],
    'Income Level': ['Low', 'Medium', 'High'],
    'Urban vs Rural Living': ['Urban', 'Rural'],
    'Depression Level': ['Low', 'Medium', 'High'],
    'Sleep Quality': ['Poor', 'Average', 'Good'],
    'Air Pollution Exposure': ['Low', 'Medium', 'High'],
    'Stress Levels': ['Low', 'Medium', 'High']
}

# Crear entradas para variables numéricas discretas
user_input = []
for feature in numeric_features:
    value = st.number_input(f"{feature}", min_value=0, step=1)
    user_input.append(value)

# Crear entradas para variables numéricas continuas
for feature in continuous_features:
    value = st.number_input(f"{feature}", value=0.0, format="%.2f")
    user_input.append(value)

# Crear entradas para variables categóricas
for feature, categories in categorical_features.items():
    value = st.selectbox(f"{feature}", categories)
    user_input.append(categories.index(value))  # Convertir a numérico

# Botón para hacer la predicción
if st.button("Predecir"): 
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    resultado = "Positivo para Alzheimer" if prediction[0] == 1 else "Negativo para Alzheimer"
    st.subheader("Resultado de la Predicción")
    st.write(resultado)



