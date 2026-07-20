from ultralytics import YOLO
import cv2

# -------- PATHS --------
MODEL_PATH = "saved_models/finalbest.pt"
IMAGE_PATH = "custom_test/image.png"

# -------- LOAD MODEL --------
model = YOLO(MODEL_PATH)

# -------- RUN PREDICTION --------
results = model(IMAGE_PATH, conf=0.25, iou=0.5)

# -------- COUNT OBJECTS --------
counts = {
    "bolt": 0,
    "locatingpin": 0,
    "nut": 0,
    "washer": 0
}

for r in results:
    for cls in r.boxes.cls:
        label = model.names[int(cls)]
        counts[label] += 1

print("\nYOLO Prediction:")
print(counts)

# -------- SAVE OUTPUT IMAGE --------
annotated = results[0].plot()
cv2.imwrite("yolo_output.png", annotated)

print("\nSaved output image as yolo_output.png")
