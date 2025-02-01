import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle 
from tensorflow.keras.datasets import boston_housing
### Titulo ## 
def main():
  st.title("Clasificación de la base de datos Boston Housing")
## Cargar el modelo ## 
def load_model():
  filename = "model_trained_regressor.pkl.gz"
  with gzip.open(filename, 'rb') as f:
    model = pickle.load(f)
  return model

