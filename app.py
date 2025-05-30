import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image
import io

from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app, resources={"r/predict": {"origins": "https://covid-19-detection-frontend.netlify.app/"}}) # <--- ADD/MODIFY THIS LINE
 # This enables CORS for all routes
# Or for specific routes: CORS(app, resources={r"/api/*": {"origins": "*"}})


# --- Configuration ---
# Set the path where your model is located relative to app.py
MODEL_PATH = 'model/covid_pypower.h5'
LABELS = ['Covid', 'Normal']
IMAGE_SIZE = (224, 224) # Model input size

# --- Load the model once when the app starts ---
# This is crucial for performance; don't load the model on every request
try:
    model = load_model(MODEL_PATH)
    print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model from {MODEL_PATH}: {e}")
    model = None # Set model to None if loading fails

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read the image file from the request
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            in_memory_file.seek(0)

            # Use PIL to open the image (more robust than cv2.imread for streams)
            pil_image = Image.open(in_memory_file).convert('RGB') # Ensure it's RGB
            frame = np.array(pil_image) # Convert PIL Image to NumPy array (OpenCV format)

            # Ensure the image is in BGR for OpenCV if needed, though PIL to RGB is usually fine
            # If your model expects BGR, uncomment below. Keras models often expect RGB.
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Preprocessing (similar to your original code)
            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, IMAGE_SIZE)
            roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB) # Convert grayscale to RGB

            # Normalize & Expand dimensions
            roi = roi_rgb.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0) # Final shape: (1, 224, 224, 3)

            # Classify the image
            print("[INFO] Classifying image...")
            preds = model.predict(roi)[0]
            label = LABELS[preds.argmax()]
            confidence = float(preds[preds.argmax()])

            print(f"[INFO] Prediction: {label} (Confidence: {confidence:.2f})")

            return jsonify({'prediction': label, 'confidence': confidence})

        except Exception as e:
            print(f"[ERROR] Error during prediction: {e}")
            return jsonify({'error': f'Error processing image: {e}'}), 500

# --- Serve the main HTML page ---
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the model directory exists
    if not os.path.exists('model'):
        os.makedirs('model')
    # Ensure the static directory exists (for web assets)
    if not os.path.exists('static'):
        os.makedirs('static')

    # For development, run with debug=True
    # For production, use a WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)