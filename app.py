import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize
from skimage.feature import hog
import joblib
import numpy as np
import cv2

# Carrega modelo
clf = joblib.load("svm_model.pkl")

st.title("Classificador de Dígitos com SVM")

# Canvas para desenhar
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button('Predict') and canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Reduz para 28x28 
    img_resized = resize(img, (28, 28), anti_aliasing=True)
    img_resized = 1.0 - img_resized  # Inverte as cores 

    # Extrair HOG
    features = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    prediction = clf.predict([features])[0]
    st.markdown(f"### 🔢 Dígito classificado: **{prediction}**")
