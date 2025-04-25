import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load MobileNetV2 model with pretrained ImageNet weights
model = MobileNetV2(weights='imagenet')

# Initialize Flask app
app = Flask(__name__)

# Health check route
@app.route('/')
def index():
    return "🚀 MobileNetV2 Object Detection API is up!"

# Unified prediction handler
def handle_prediction():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Convert image to OpenCV format
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    # Resize and preprocess for MobileNetV2
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0]

    # Format result
    result = [
        {"label": label, "description": desc, "probability": float(prob)}
        for (label, desc, prob) in decoded
    ]

    return jsonify({"predictions": result})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    return handle_prediction()

# Optional /upload endpoint (aliases /predict)
@app.route('/upload', methods=['POST'])
def upload():
    return handle_prediction()

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
