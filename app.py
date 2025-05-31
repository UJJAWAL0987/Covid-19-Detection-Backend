import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image
import io
from flask_cors import CORS

# For MongoDB
from pymongo import MongoClient
# For local .env file management (remove in production if using platform env vars)
from dotenv import load_dotenv

# Load environment variables from .env file for local development
# In production (Fly.io/Render), these will be set directly as secrets/env vars
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='static')

# CORRECTED CORS CONFIGURATION (use your Netlify URL)
CORS(app, resources={r"/predict": {"origins": ["https://covid-19-detection-frontend.netlify.app"]}})

# --- MongoDB Configuration ---
# Get MongoDB URI from environment variable
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "covid_detection_db" # Name your database
COLLECTION_NAME = "predictions" # Name your collection

# Initialize MongoDB client
mongo_client = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[DB_NAME]
        predictions_collection = db[COLLECTION_NAME]
        print(f"[INFO] MongoDB connected to database: {DB_NAME}")
    except Exception as e:
        print(f"[ERROR] Could not connect to MongoDB: {e}")
        mongo_client = None # Set to None if connection fails
else:
    print("[WARNING] MONGO_URI environment variable not set. MongoDB will not be used.")


# --- Configuration ---
MODEL_PATH = 'model/covid_pypower.h5'
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
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            in_memory_file.seek(0)

            pil_image = Image.open(in_memory_file).convert('RGB')
            frame = np.array(pil_image)

            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, IMAGE_SIZE)
            roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

            roi = roi_rgb.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            print("[INFO] Classifying image...")
            preds = model.predict(roi)[0]
            label = LABELS[preds.argmax()]
            confidence = float(preds[preds.argmax()])

            print(f"[INFO] Prediction: {label} (Confidence: {confidence:.2f})")

            # --- Store prediction in MongoDB ---
            if mongo_client:
                try:
                    prediction_data = {
                        "prediction": label,
                        "confidence": confidence,
                        "timestamp": datetime.now() # Requires `from datetime import datetime` at top
                    }
                    predictions_collection.insert_one(prediction_data)
                    print("[INFO] Prediction stored in MongoDB.")
                except Exception as e:
                    print(f"[ERROR] Failed to store prediction in MongoDB: {e}")

            return jsonify({'prediction': label, 'confidence': confidence})

        except Exception as e:
            print(f"[ERROR] Error during prediction: {e}")
            return jsonify({'error': f'Error processing image: {e}'}), 500

# --- Serve the main HTML page ---
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True, host='0.0.0.0', port=5000)