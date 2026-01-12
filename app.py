import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌱 Plant Disease Detection System")

model = tf.keras.models.load_model("plant_disease_model.h5")

class_names = list(model.class_names) if hasattr(model, "class_names") else [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Cherry___Powdery_mildew","Cherry___healthy",
    "Corn___Cercospora_leaf_spot","Corn___Common_rust","Corn___Northern_Leaf_Blight","Corn___healthy",
    "Grape___Black_rot","Grape___Esca","Grape___Leaf_blight","Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Soybean___healthy",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

uploaded = st.file_uploader("Upload a leaf image", type=["jpg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img = img.resize((128,128))
    st.image(img, caption="Uploaded Leaf")

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    idx = np.argmax(prediction)

    st.success(f"Predicted Disease: {class_names[idx]}")
    st.info(f"Confidence: {np.max(prediction)*100:.2f}%")
