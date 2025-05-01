import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load('model.pkl')

# Fungsi preprocessing dan prediksi
def preprocess_and_predict(image: Image.Image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    resized = cv2.resize(edges, (64, 64)).flatten().reshape(1, -1)
    pred = model.predict(resized)[0]
    prob = model.predict_proba(resized)[0][pred]
    return pred, prob, edges

# UI
st.title("Rust Detection using Canny")
st.write("Upload a metal surface image to check for rust.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        pred, prob, edge_img = preprocess_and_predict(image)

    st.subheader("Result")
    if pred == 1:
        st.error(f"🛑 Rust Detected ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Clean Surface ({prob*100:.2f}%)")

    st.subheader("Canny Edge Result")
    st.image(edge_img, caption="Edge Detection Output", use_column_width=True, clamp=True)
