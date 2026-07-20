import os
import pandas as pd
from ultralytics import YOLO

# ---------------- PATHS ----------------
MODEL_PATH = "saved_models/best.pt"
TRAIN_IMG_DIR = "train/train"
LABELS_CSV = "train_labels.csv"
OUT_CSV = "yolo_features_train.csv"

# class order must match submission
CLASSES = ["bolt", "locatingpin", "nut", "washer"]

model = YOLO(MODEL_PATH)
labels_df = pd.read_csv(LABELS_CSV)

rows = []

for _, row in labels_df.iterrows():
    img_name = row["image_name"]
    img_path = os.path.join(TRAIN_IMG_DIR, img_name)

    results = model(img_path, conf=0.25, iou=0.5, verbose=False)

    # YOLO counts
    counts = {c: 0 for c in CLASSES}
    confs = []
    areas = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls)
            if cls_id < len(CLASSES):
                counts[CLASSES[cls_id]] += 1
                confs.append(float(box.conf))
                xyxy = box.xyxy[0]
                areas.append((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))

    rows.append({
        "image_name": img_name,
        "bolt_yolo": counts["bolt"],
        "locatingpin_yolo": counts["locatingpin"],
        "nut_yolo": counts["nut"],
        "washer_yolo": counts["washer"],
        "total_boxes": sum(counts.values()),
        "avg_conf": sum(confs)/len(confs) if confs else 0,
        "total_area": sum(areas) if areas else 0,

        # ground truth
        "bolt_gt": row["bolt"],
        "locatingpin_gt": row["locatingpin"],
        "nut_gt": row["nut"],
        "washer_gt": row["washer"],
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("✅ YOLO feature training CSV created")
