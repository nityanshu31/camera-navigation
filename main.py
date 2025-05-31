from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

app = FastAPI()

model = YOLO("yolov8n.pt")

def detect_direction(frame):
    results = model(frame)[0]
    width = frame.shape[1]
    left_zone = width * 0.33
    right_zone = width * 0.66
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

    if directions["center"]:
        return "❌ Stop, obstacle ahead"
    elif directions["left"] and not directions["right"]:
        return "➡️ Go right"
    elif directions["right"] and not directions["left"]:
        return "⬅️ Go left"
    else:
        return "✅ Path is clear"

@app.get("/")
def root():
    return {"message": "Camera Navigation API is running"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    direction = detect_direction(frame)
    return JSONResponse({"direction": direction})
