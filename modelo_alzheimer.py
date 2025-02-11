import streamlit as st
import numpy as np
import gzip
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cargar modelo con cacheo
@st.cache_resource
def load_model():
    filename = "mejor_modelo_redes.pkl.gz"
    try:
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Cargar los LabelEncoders
@st.cache_resource
def load_label_encoders():
    encoder_file = "label_encoders.pkl"
    try:
        with open(encoder_file, "rb") as f:
            encoders = pickle.load(f)
        return encoders
    except Exception as e:
        st.error(f"Error al cargar los encoders: {e}")
        return None

# Cargar modelo y encoders
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

# Obtener valores numéricos
for feature in numeric_features:
    user_input[feature] = st.number_input(feature, min_value=0, step=1, format="%d")

for feature in continuous_features:
    user_input[feature] = st.number_input(feature, value=0.0, format="%.2f")

# Obtener valores categóricos
for feature in categorical_features:
    if label_encoders and feature in label_encoders:
        user_input[feature] = st.selectbox(feature, label_encoders[feature].classes_)
    else:
        user_input[feature] = st.text_input(f"Ingrese un valor para {feature}")

# Botón de predicción
if st.button("Predecir"):
    if model is None or label_encoders is None:
        st.error("No se puede realizar la predicción porque el modelo o los encoders no se cargaron correctamente.")
    else:
        try:
            df_input = pd.DataFrame([user_input])

            # Aplicar Label Encoding
            for col in categorical_features:
                if col in label_encoders:
                    if user_input[col] in label_encoders[col].classes_:
                        df_input[col] = label_encoders[col].transform([user_input[col]])[0]
                    else:
                        st.error(f"El valor '{user_input[col]}' no está en el conjunto de entrenamiento del LabelEncoder.")
                        st.stop()

            # Convertir a float para evitar problemas con el modelo
            df_input[numeric_features + continuous_features] = df_input[numeric_features + continuous_features].astype(np.float32)

            # Verificar entrada transformada
            st.write("📊 *Datos de entrada transformados:*", df_input)

            # Convertir a array para el modelo
            input_array = df_input.to_numpy().reshape(1, -1)

            # Hacer predicción
            prediction = model.predict(input_array)
            
            # Si el modelo usa probabilidades, ajustamos la interpretación
            if hasattr(model, "predict_proba"):  
                probabilidad = model.predict_proba(input_array)[0][1]  # Probabilidad de Alzheimer
                st.write(f"🔍 *Probabilidad de Alzheimer:* {probabilidad:.4f}")
                resultado = "Positivo para Alzheimer" if probabilidad > 0.5 else "Negativo para Alzheimer"
            else:
                resultado = "Positivo para Alzheimer" if prediction[0] == 1 else "Negativo para Alzheimer"

            # Mostrar resultado
            st.subheader("🧠 *Resultado de la Predicción*")
            st.write(f"*{resultado}*")

        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")

