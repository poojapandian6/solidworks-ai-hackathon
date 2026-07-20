import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# ---------------- PATHS ----------------
IMAGES_DIR = os.path.join("train", "train")   # where original images are
BBOX_CSV = "train_bboxes.csv"
YOLO_DIR = "yolo_dataset"

# ---------------- CLASS MAPPING ----------------
# MUST match everywhere
CLASS_MAP = {
    "nut": 0,
    "bolt": 1,
    "locatingpin": 2,
    "washer": 3
}

# ---------------- CREATE DIRECTORIES ----------------
for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)

# ---------------- LOAD CSV ----------------
df = pd.read_csv(BBOX_CSV)

# ---------------- TRAIN / VAL SPLIT (by image) ----------------
unique_images = df["image_name"].unique()

train_imgs, val_imgs = train_test_split(
    unique_images, test_size=0.2, random_state=42
)

# ---------------- BBOX CONVERSION ----------------
def convert_bbox(row, img_w, img_h):
    x_center = ((row["x_min"] + row["x_max"]) / 2) / img_w
    y_center = ((row["y_min"] + row["y_max"]) / 2) / img_h
    width = (row["x_max"] - row["x_min"]) / img_w
    height = (row["y_max"] - row["y_min"]) / img_h
    return x_center, y_center, width, height

# ---------------- PROCESS EACH IMAGE ----------------
for img_id in unique_images:
    split = "train" if img_id in train_imgs else "val"

    # actual filename
    img_filename = img_id

    img_path = os.path.join(IMAGES_DIR, img_filename)

    # open image to get size
    img = Image.open(img_path)
    img_w, img_h = img.size

    # copy image
    dst_img_path = os.path.join(YOLO_DIR, "images", split, img_filename)
    if not os.path.exists(dst_img_path):
        img.save(dst_img_path)

    # create label file
    label_path = os.path.join(
        YOLO_DIR, "labels", split, img_filename.replace(".png", ".txt")
    )

    with open(label_path, "w") as f:
        rows = df[df["image_name"] == img_id]
        for _, row in rows.iterrows():
            class_id = CLASS_MAP[row["class"]]
            xc, yc, w, h = convert_bbox(row, img_w, img_h)
            f.write(f"{class_id} {xc} {yc} {w} {h}\n")

print("✅ YOLO dataset preparation complete.")
