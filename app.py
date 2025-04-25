import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the MobileNetV2 model pre-trained on ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def prepare_image(image):
    # Resize image to 224x224, the input size expected by MobileNetV2
    image_resized = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_array = np.expand_dims(image_rgb, axis=0)
    image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_preprocessed

@app.route('/')
def hello():
    return 'Server is up!'

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get image data from the request
    image_data = request.data
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Prepare the image for MobileNetV2
    image_preprocessed = prepare_image(img)

    # Make predictions
    predictions = model.predict(image_preprocessed)

    # Decode predictions to human-readable labels
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)

    results = [
        {"label": pred[1], "probability": float(pred[2])}
        for pred in decoded_predictions[0]
    ]

    return jsonify({"detected_objects": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # fallback to 5000 for local testing
    app.run(host='0.0.0.0', port=port)
