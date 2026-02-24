import os
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms

REFERENCE_IMAGES_DIR = "reference_images"  
REFERENCE_LOG_PATH = Path("reference_log.json")
MODEL_PATH = "model0.pt"

model = YOLO(MODEL_PATH)
IMAGE_SIZE = (640, 640)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

def get_image_stats(image):
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np]*3, axis=-1)
    width, height = image.width, image.height
    mean_pixel_value = float(np.mean(image_np))
    std_pixel_value = float(np.std(image_np))
    color_hist = {}
    for i, color in enumerate(['r', 'g', 'b']):
        hist, _ = np.histogram(image_np[..., i], bins=16, range=(0, 255), density=True)
        color_hist[color] = hist.tolist()
    return {
        "width": width,
        "height": height,
        "mean_pixel_value": mean_pixel_value,
        "std_pixel_value": std_pixel_value,
        "color_histogram": color_hist
    }

def get_predictions(image_tensor, model):
    with torch.no_grad():
        pred = model(image_tensor)
    predictions = []
    for i in range(len(pred[0].boxes.cls)):
        curr = dict()
        curr["label"] = "timmies" if pred[0].names[int(pred[0].boxes.cls[i])] == "Timmies" else "paper_cup"
        curr["confidence"] = round(float(pred[0].boxes.conf[i]), 2)
        curr["bbox"] = [round(x) for x in pred[0].boxes.xywh[i].tolist()]
        predictions.append(curr)
    return predictions

log = []
for fname in os.listdir(REFERENCE_IMAGES_DIR):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(REFERENCE_IMAGES_DIR, fname)
        image = Image.open(image_path)
        image_stats = get_image_stats(image)
        image_tensor = transform(image).unsqueeze(0)
        predictions = get_predictions(image_tensor, model)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input_image_stats": image_stats,
            "predictions": predictions
        }
        log.append(entry)

with open(REFERENCE_LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2)

print(f"Reference log created with {len(log)} entries at {REFERENCE_LOG_PATH}")