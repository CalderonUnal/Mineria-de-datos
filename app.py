import streamlit as st
from PIL import Image
import numpy as np
import gzip
import pickle
import tensorflow as tf

def preprocess_image(image):
    """ Preprocesa la imagen para que tenga el formato correcto para la clasificación MNIST."""
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Normalizar
    image_array = image_array.flatten().reshape(1, -1)  # Convertir a vector de 784 elementos
    return image_array

@st.cache_resource
def load_model():
    """ Carga el modelo previamente entrenado."""
    filename = "model_trained_classifier.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el modelo entrenado
model = load_model()

# Hiperparámetros del modelo
hyperparameters = {
    "optimizer": "adam",
    "loss_function": "sparse_categorical_crossentropy",
    "epochs": 10,
    "batch_size": 32
}

# Interfaz en Streamlit
st.title("Clasificación de Dígitos - MNIST")
st.write("Sube una imagen de un número escrito a mano para clasificarlo.")

uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    preprocessed_image = preprocess_image(image)
    
    if st.button("Clasificar Imagen"):
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        st.success(f"La imagen fue clasificada como: {predicted_class}")

# Mostrar hiperparámetros del modelo
st.subheader("Hiperparámetros del Mejor Modelo")
st.json(hyperparameters)

