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

@st.cache_resource
def load_label_encoders():
    encoder_file = "label_encoders.pkl"
    with open(encoder_file, "rb") as f:
        encoders = pickle.load(f)
    return encoders

model = load_model()
label_encoders = load_label_encoders()

st.title("Predicción de Alzheimer")

# Definir características categóricas y numéricas
categorical_features = [
    'Country', 'Gender', 'Smoking Status', 'Alcohol Consumption', 'Diabetes',
    'Hypertension', 'Cholesterol Level', 'Family History of Alzheimer’s',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Urban vs Rural Living', 'Physical Activity Level', 'Depression Level',
    'Sleep Quality', 'Dietary Habits', 'Air Pollution Exposure',
    'Social Engagement Level', 'Income Level', 'Stress Levels'
]

numeric_features = ['Age', 'Education Level', 'Cognitive Test Score']
continuous_features = ['BMI']
user_input = {}

# Obtener valores de entrada numéricos
def get_numeric_input(label, min_value=0, step=1):
    return st.number_input(label, min_value=min_value, step=step)

def get_continuous_input(label):
    return st.number_input(label, value=0.0, format="%.2f")

for feature in numeric_features:
    user_input[feature] = get_numeric_input(feature)

for feature in continuous_features:
    user_input[feature] = get_continuous_input(feature)

# Obtener valores de entrada categóricos
def get_categorical_input(label, options):
    return st.selectbox(label, options)

for feature in categorical_features:
    if feature in label_encoders:
        user_input[feature] = get_categorical_input(feature, label_encoders[feature].classes_)

if st.button("Predecir"):
    if model is None:
        st.error("No se puede realizar la predicción porque el modelo no se cargó correctamente.")
    else:
        try:
            df_input = pd.DataFrame([user_input])

            # Aplicar Label Encoding usando los LabelEncoders guardados
            for col in categorical_features:
                if col in label_encoders:
                    df_input[col] = label_encoders[col].transform(df_input[col])
            
            # Convertir a array NumPy para el modelo
            input_array = df_input.to_numpy()
            prediction = model.predict(input_array)

            resultado = "Positivo para Alzheimer" if prediction[0] == 1 else "Negativo para Alzheimer"
            st.subheader("Resultado de la Predicción")
            st.write(resultado)
        except Exception as e:
            st.error(f"Ocurrió un error al hacer la predicción: {str(e)}")

