import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load('model.pkl')

# Fungsi preprocessing dan prediksi
def preprocess_and_predict(image: Image.Image):
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original = img_cv.copy()

    # Preprocessing
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Resize dan flatten untuk prediksi
    resized = cv2.resize(edges, (64, 64)).flatten().reshape(1, -1)
    pred = model.predict(resized)[0]
    prob = model.predict_proba(resized)[0][pred]

    # Temukan kontur dari hasil edge
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = original.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # green contour

    # Convert BGR ke RGB untuk Streamlit
    contour_img_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)

    return pred, prob, edges, contour_img_rgb

# Streamlit UI
st.set_page_config(page_title="Rust Detection", layout="centered")
st.title("🔎 Rust Detection using Canny")
st.write("Upload a metal surface image to check for rust.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        pred, prob, edge_img, contour_img = preprocess_and_predict(image)

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"🛑 Rust Detected ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Clean Surface ({prob*100:.2f}%)")

    # Tampilkan hasil
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Canny Edge")
        st.image(edge_img, clamp=True, use_column_width=True)

    with col2:
        st.subheader("Contours on Image")
        st.image(contour_img, use_column_width=True)
