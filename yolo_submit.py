import os
import pandas as pd
from ultralytics import YOLO

# ---------------- PATHS ----------------
TEST_DIR = os.path.join("test", "test")
MODEL_PATH = "saved_models/finalbest.pt"

OUTPUT_CSV = "submission_yolo.csv"

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- CLASS MAP ----------------
# MUST match data.yaml used during training
CLASS_MAP = {
    0: "nut",
    1: "bolt",
    2: "locatingpin",
    3: "washer"
}

rows = []

# ---------------- RUN INFERENCE ----------------
for img_name in sorted(os.listdir(TEST_DIR)):
    if not img_name.lower().endswith(".png"):
        continue

    img_path = os.path.join(TEST_DIR, img_name)

    results = model(
        img_path,
        conf=0.30,
        iou=0.5,
        verbose=False
    )

    # initialize counts
    counts = {
        "bolt": 0,
        "locatingpin": 0,
        "nut": 0,
        "washer": 0
    }

    # count detections
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls)
            if cls_id in CLASS_MAP:
                class_name = CLASS_MAP[cls_id]
                counts[class_name] += 1

    # append row in REQUIRED CSV ORDER
    rows.append([
        img_name,
        counts["bolt"],
        counts["locatingpin"],
        counts["nut"],
        counts["washer"]
    ])

# ---------------- SAVE CSV ----------------
df = pd.DataFrame(
    rows,
    columns=["image_name", "bolt", "locatingpin", "nut", "washer"]
)

df.to_csv(OUTPUT_CSV, index=False)

print("✅ YOLO submission CSV created successfully")
print("📄 File:", OUTPUT_CSV)
print("🧮 Total images:", len(df))
