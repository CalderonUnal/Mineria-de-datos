import streamlit as st
import numpy as np
import gzip
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
st.title("Predicción de Alzheimer")

# Variables del dataset
categorical_features = {
    'Country': ['Australia', 'Brazil', 'Canada', 'China', 'France', 'Germany', 'India', 'Italy', 'Japan', 'Mexico', 
                'Norway', 'Russia', 'Saudi Arabia', 'South Africa', 'South Korea', 'Spain', 'Sweden', 'UK', 'USA'],
    'Gender': ['Male', 'Female'],
    'Physical Activity Level': ['Low', 'Medium', 'High'],
    'Smoking Status': ['Never', 'Former', 'Current'],
    'Alcohol Consumption': ['Never', 'Occasionally', 'Regularly'],
    'Diabetes': ['No', 'Yes'],
    'Hypertension': ['No', 'Yes'],
    'Cholesterol Level': ['Low', 'Normal', 'High'],
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

numeric_features = ['Age', 'Education Level', 'Cognitive Test Score']
continuous_features = ['BMI']

# Entrada de usuario
user_input = []

# Entradas para variables numéricas discretas
for feature in numeric_features:
    value = st.number_input(f"{feature}", min_value=0, step=1)
    user_input.append(value)

# Entradas para variables numéricas continuas
for feature in continuous_features:
    value = st.number_input(f"{feature}", value=0.0, format="%.2f")
    user_input.append(value)

# Entradas para variables categóricas
for feature, categories in categorical_features.items():
    value = st.selectbox(f"{feature}", categories)
    user_input.append(value)

# Botón para hacer la predicción
if st.button("Predecir"):
    if model is None:
        st.error("No se puede realizar la predicción porque el modelo no se cargó correctamente.")
    else:
        try:
            # Convertir lista de entrada a DataFrame
            column_names = ['Country', 'Age', 'Gender', 'Education Level', 'BMI',
                            'Physical Activity Level', 'Smoking Status', 'Alcohol Consumption',
                            'Diabetes', 'Hypertension', 'Cholesterol Level',
                            'Family History of Alzheimer’s', 'Cognitive Test Score',
                            'Depression Level', 'Sleep Quality', 'Dietary Habits',
                            'Air Pollution Exposure', 'Employment Status', 'Marital Status',
                            'Genetic Risk Factor (APOE-ε4 allele)', 'Social Engagement Level',
                            'Income Level', 'Stress Levels', 'Urban vs Rural Living']

            df_input = pd.DataFrame([user_input], columns=column_names)

            # Aplicar One-Hot Encoding sin eliminar la primera categoría
            categorical_columns = ['Country', 'Gender', 'Smoking Status', 'Alcohol Consumption', 'Diabetes',
                                   'Hypertension', 'Cholesterol Level', 'Family History of Alzheimer’s',
                                   'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
                                   'Urban vs Rural Living']

            df_input = pd.get_dummies(df_input, columns=categorical_columns, drop_first=False)

            # Variables ordinales
            ordinal_columns = ['Physical Activity Level', 'Depression Level', 'Sleep Quality', 'Dietary Habits',
                               'Air Pollution Exposure', 'Social Engagement Level', 'Income Level', 'Stress Levels',
                               'Education Level']

            # Aplicar Label Encoding
            label_encoders = {col: LabelEncoder() for col in ordinal_columns}
            
            for col in ordinal_columns:
                df_input[col] = label_encoders[col].fit_transform(df_input[col])

            # Asegurar que las columnas sean las mismas que en el modelo entrenado
            expected_columns = model.input_shape[1]
            if df_input.shape[1] != expected_columns:
                st.error(f"Error: Se esperaban {expected_columns} columnas, pero se obtuvieron {df_input.shape[1]}.")
                st.write("Columnas actuales:", df_input.columns.tolist())
            else:
                # Convertir a NumPy array y predecir
                input_array = df_input.to_numpy()
                prediction = model.predict(input_array)

                resultado = "Positivo para Alzheimer" if prediction[0] == 1 else "Negativo para Alzheimer"
                st.subheader("Resultado de la Predicción")
                st.write(resultado)
        except Exception as e:
            st.error(f"Ocurrió un error al hacer la predicción: {str(e)}")
