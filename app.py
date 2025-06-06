import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image
import io

from flask_cors import CORS

app = Flask(__name__)

from dotenv import load_dotenv # Import to load environment variables

# Load environment variables from .env file (important for production)
load_dotenv()

# CORRECTED CORS CONFIGURATION
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://covid-19-detection-frontend.netlify.app")
CORS(app, resources={r'/predict': {"origins": [FRONTEND_ORIGIN]}})


# --- Configuration ---
# MODEL_PATH needs to be relative to the app.py file when deployed.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'covid_pypower.h5')
LABELS = ['Covid', 'Normal']
LABELS = ['Covid', 'Normal']
IMAGE_SIZE = (224, 224) # Model input size

# --- Load the model once when the app starts ---
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
            # Assuming your model expects RGB input for 224x224x3
            # If your model was trained on grayscale, you might need to adjust.
            # The current approach converts to grayscale then back to RGB; review your model's training data for consistency.
            # If your model expects RGB directly, you can simply resize `pil_image` and convert to array.
            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # This line implies frame is BGR initially, which it might not be from PIL. If issues, check this.
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