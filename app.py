import os
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Server is up!'

@app.route('/upload', methods=['POST'])
def upload_image():
    image_data = request.data
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Object detection - example: face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = [
        {"label": "face", "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        for (x, y, w, h) in faces
    ]

    return jsonify({"detected_objects": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # fallback to 5000 for local testing
    app.run(host='0.0.0.0', port=port)
