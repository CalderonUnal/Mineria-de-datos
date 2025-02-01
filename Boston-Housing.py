import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle 
from tensorflow.keras.datasets import boston_housing
## Cargar el modelo ## 
def load_model():
  filename = "model_trained_regressor.pkl.gz"
  with gzip.open(filename, 'rb') as f:
    model = pickle.load(f)
  return model

