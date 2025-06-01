import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image
import io

# --- Configuration ---
MODEL_PATH = 'covid_pypower.h5'  # Model should be in the same directory or accessible path
LABELS = ['Covid', 'Normal']
IMAGE_SIZE = (224, 224)

# --- Load the model once when the app starts ---
# This uses st.cache_resource to load the model only once, improving performance
@st.cache_resource
def load_my_model():
    try:
        model = load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

model = load_my_model()

# --- Streamlit App UI ---
st.title("COVID-19 Chest X-Ray Detection")
st.write("Upload a chest X-ray image to get a prediction.")

if model is None:
    st.warning("Prediction functionality is unavailable because the model could not be loaded.")
else:
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded X-Ray', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        try:
            # Read the image file from the request
            in_memory_file = io.BytesIO(uploaded_file.getvalue())

            # Use PIL to open the image
            pil_image = Image.open(in_memory_file).convert('RGB')
            frame = np.array(pil_image)

            # Preprocessing (similar to your original Flask code)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            roi_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, IMAGE_SIZE)
            roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

            # Normalize & Expand dimensions
            roi = roi_rgb.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)  # Final shape: (1, 224, 224, 3)

            # Make prediction
            preds = model.predict(roi)[0]
            label = LABELS[preds.argmax()]
            confidence = float(preds[preds.argmax()])

            st.success(f"**Prediction:** {label} (Confidence: {confidence:.2f})")

        except Exception as e:
            st.error(f"Error processing image: {e}")