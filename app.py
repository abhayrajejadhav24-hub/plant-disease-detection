import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Plant Disease Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

st.title("Plant Disease Detection")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    st.success(f"Prediction class index: {class_index}")
