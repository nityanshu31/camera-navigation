from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load YOLO model once and force CPU to avoid GPU/Render memory issues
model = YOLO("yolov8n.pt").to("cpu")

def detect_direction(frame):
    width = frame.shape[1]
    left_zone = width * 0.33
    right_zone = width * 0.66

    results = model(frame, verbose=False)[0]

    directions = {"left": False, "center": False, "right": False}

    for box in results.boxes:
        x1, _, x2, _ = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        if center_x < left_zone:
            directions["left"] = True
        elif center_x > right_zone:
            directions["right"] = True
        else:
            directions["center"] = True

    if directions["left"] and directions["center"] and directions["right"]:
        return "❌ Obstacle ahead, no way to go"
    elif directions["center"]:
        if directions["left"] and not directions["right"]:
            return "➡️ Go right"
        elif directions["right"] and not directions["left"]:
            return "⬅️ Go left"
        elif not directions["left"] and not directions["right"]:
            return "🔄 Slight obstacle ahead, avoid center"
        else:
            return "✅ Path is clear"
    else:
        return "✅ Path is clear"

@app.route('/')
def home():
    return "✅ YOLOv8 Obstacle Navigation API is running"

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((640, 480))  # Resize to reduce memory usage
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        direction = detect_direction(frame)
        return jsonify({"direction": direction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
