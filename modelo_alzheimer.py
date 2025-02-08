import streamlit as st
import numpy as np
import gzip
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    filename = "mejor_modelo_redes.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Predicción de Alzheimer")

# Variables categóricas
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
user_input = []

for feature in numeric_features:
    value = st.number_input(f"{feature}", min_value=0, step=1)
    user_input.append(value)

for feature in continuous_features:
    value = st.number_input(f"{feature}", value=0.0, format="%.2f")
    user_input.append(value)

for feature, categories in categorical_features.items():
    value = st.selectbox(f"{feature}", categories)
    user_input.append(value)

if st.button("Predecir"):
    if model is None:
        st.error("No se puede realizar la predicción porque el modelo no se cargó correctamente.")
    else:
        try:
            column_names = ['Country', 'Age', 'Gender', 'Education Level', 'BMI',
                            'Physical Activity Level', 'Smoking Status', 'Alcohol Consumption',
                            'Diabetes', 'Hypertension', 'Cholesterol Level',
                            'Family History of Alzheimer’s', 'Cognitive Test Score',
                            'Depression Level', 'Sleep Quality', 'Dietary Habits',
                            'Air Pollution Exposure', 'Employment Status', 'Marital Status',
                            'Genetic Risk Factor (APOE-ε4 allele)', 'Social Engagement Level',
                            'Income Level', 'Stress Levels', 'Urban vs Rural Living']

            df_input = pd.DataFrame([user_input], columns=column_names)

            categorical_columns = ['Country', 'Gender', 'Smoking Status', 'Alcohol Consumption', 'Diabetes',
                                   'Hypertension', 'Cholesterol Level', 'Family History of Alzheimer’s',
                                   'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
                                   'Urban vs Rural Living']

            # Aplicar One-Hot Encoding
            df_input = pd.get_dummies(df_input, columns=categorical_columns, drop_first=True)

            # Aplicar Label Encoding a variables ordinales
            ordinal_columns = ['Physical Activity Level', 'Depression Level', 'Sleep Quality', 'Dietary Habits',
                               'Air Pollution Exposure', 'Social Engagement Level', 'Income Level', 'Stress Levels',
                               'Education Level']

            label_encoders = {col: LabelEncoder() for col in ordinal_columns}
            for col in ordinal_columns:
                df_input[col] = label_encoders[col].fit_transform(df_input[col])

            # Verificación de número de columnas
            expected_columns = [...]  # Lista de las 46 columnas esperadas en el modelo

            if df_input.shape[1] != 46:
                st.error(f"Error: Se esperaban 46 columnas, pero se obtuvieron {df_input.shape[1]}.")
                st.write("Columnas actuales después de transformación:", df_input.columns.tolist())

                missing_cols = [col for col in expected_columns if col not in df_input.columns]
                extra_cols = [col for col in df_input.columns if col not in expected_columns]

                st.write("Columnas extra:", extra_cols)
                st.write("Columnas faltantes:", missing_cols)

                # Solución: Reindexar para asegurar que todas las columnas estén presentes
                df_input = df_input.reindex(columns=expected_columns, fill_value=0)
                st.write(f"Después de la corrección, total de columnas: {df_input.shape[1]}")

            if df_input.shape[1] == 46:
                # Convertir a array NumPy para el modelo
                input_array = df_input.to_numpy()
                prediction = model.predict(input_array)

                resultado = "Positivo para Alzheimer" if prediction[0] == 1 else "Negativo para Alzheimer"
                st.subheader("Resultado de la Predicción")
                st.write(resultado)
        except Exception as e:
            st.error(f"Ocurrió un error al hacer la predicción: {str(e)}")

