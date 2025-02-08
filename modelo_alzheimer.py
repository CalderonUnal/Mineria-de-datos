import streamlit as st
import numpy as np
import gzip
import pickle
import pandas as pd

# Función para cargar el modelo
@st.cache_resource
def load_model():
    filename = "mejor_modelo_redes.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    if not hasattr(model, 'predict'):
        st.error("El modelo cargado no es válido. Asegúrate de que es un modelo de clasificación con el método 'predict'.")
        return None
    return model

# Cargar el modelo entrenado
model = load_model()

# Título de la aplicación
st.title("Predicción Alzheimer")

# Definir las características del dataset
numeric_features = ['Age', 'Education Level', 'BMI', 'Cognitive Test Score']
ordinal_features = ['Physical Activity Level', 'Depression Level', 'Sleep Quality', 'Dietary Habits',
                    'Air Pollution Exposure', 'Social Engagement Level', 'Income Level', 'Stress Levels']

dummy_features = ['Country', 'Gender', 'Smoking Status', 'Alcohol Consumption', 'Diabetes', 'Hypertension',
                  'Cholesterol Level', 'Family History of Alzheimer’s', 'Employment Status', 'Marital Status',
                  'Genetic Risk Factor (APOE-ε4 allele)', 'Urban vs Rural Living']

# Crear entradas para variables numéricas
data_input = {}
for feature in numeric_features:
    data_input[feature] = st.number_input(f"{feature}", min_value=0.0, format="%.2f")

# Crear entradas para variables ordinales
ordinal_mappings = {
    'Physical Activity Level': ['Low', 'Medium', 'High'],
    'Depression Level': ['Low', 'Medium', 'High'],
    'Sleep Quality': ['Poor', 'Average', 'Good'],
    'Dietary Habits': ['Unhealthy', 'Average', 'Healthy'],
    'Air Pollution Exposure': ['Low', 'Medium', 'High'],
    'Social Engagement Level': ['Low', 'Medium', 'High'],
    'Income Level': ['Low', 'Medium', 'High'],
    'Stress Levels': ['Low', 'Medium', 'High']
}

for feature, categories in ordinal_mappings.items():
    data_input[feature] = categories.index(st.selectbox(f"{feature}", categories))

# Crear entradas para variables dummy
for feature in dummy_features:
    categories = pd.unique(df[feature])
    selected_category = st.selectbox(f"{feature}", categories)
    for category in categories:
        column_name = f"{feature}_{category}"
        data_input[column_name] = 1 if selected_category == category else 0

# Botón para hacer la predicción
if st.button("Predecir"):
    if model is None:
        st.error("No se puede realizar la predicción porque el modelo no se cargó correctamente.")
    else:
        try:
            input_array = np.array([list(data_input.values())]).astype(float)
            prediction = model.predict(input_array)
            resultado = "Positivo para Alzheimer" if prediction[0] == 1 else "Negativo para Alzheimer"
            st.subheader("Resultado de la Predicción")
            st.write(resultado)
        except Exception as e:
            st.error(f"Ocurrió un error al hacer la predicción: {str(e)}")
