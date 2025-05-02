import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

st.title("Rust Recognition App")
st.write("Upload a metal surface image to detect rust visually.")

model = joblib.load("model.pkl")

def extract_features(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_rust = np.array([5, 50, 50])
    upper_rust = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_rust, upper_rust)
    resized = cv2.resize(mask, (64, 64)).flatten().reshape(1, -1)
    return resized, mask, img_cv

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        features, mask, original_cv = extract_features(image)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]

        st.subheader("Prediction")
        if pred == 1:
            st.error(f"🛑 Rust Detected ({prob*100:.2f}%)")

            # Draw contours of rust areas
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original_cv, contours, -1, (0, 255, 0), 2)
            st.image(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB), caption="Detected Rust Areas", use_column_width=True)
        else:
            st.success(f"✅ Clean Surface ({prob*100:.2f}%)")

        st.subheader("Rust Color Mask")
        st.image(mask, clamp=True, use_column_width=True)
