# 🔩 AI-Based Fastener Detection for SolidWorks Assemblies using YOLOv8

An AI-powered computer vision project that detects and classifies mechanical fasteners from SolidWorks assembly images using the YOLOv8 object detection model. The system automatically identifies components such as **Bolts, Nuts, Washers, and Locating Pins**, enabling faster and more reliable inspection of mechanical assemblies.

---

## 📌 Overview

Mechanical assembly inspection is often performed manually, making it time-consuming and susceptible to human error. This project leverages **YOLOv8** to automate the detection and counting of common fasteners in SolidWorks assembly images.

The trained model predicts the location and class of each fastener and generates an annotated output image for visualization.

---

## ✨ Features

- Detects multiple fastener types in a single image
- YOLOv8-based object detection
- Automatic bounding box generation
- Fast inference on new images
- Generates annotated output images
- Simple prediction pipeline

---

## 🛠️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy
- Pandas

---

## 📂 Project Structure

```
solidworks-ai-hackathon/
│
├── custom_test/
│   └── image.png
│
├── saved_models/
│   └── finalbest.pt
│
├── prepare_yolo_data.py
├── build_yolo_features.py
├── clean_yolo_features.py
├── predict_yolo.py
├── yolo_submit.py
│
├── sample_submission.csv
├── yolo_output.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/poojapandian6/solidworks-ai-hackathon.git

cd solidworks-ai-hackathon
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 📦 Model

The trained YOLO model should be placed inside:

```
saved_models/
```

Expected file:

```
saved_models/
    finalbest.pt
```

> If the model file is not included in this repository, place your trained model in the folder above before running inference.

---

## 🚀 Running Prediction

Place the test image inside:

```
custom_test/image.png
```

Run:

```bash
python predict_yolo.py
```

Example output:

```
YOLO Prediction:
{
    "bolt": 0,
    "locatingpin": 0,
    "nut": 1,
    "washer": 0
}

Saved output image as yolo_output.png
```

---

## 🎯 Supported Classes

- Bolt
- Nut
- Washer
- Locating Pin

---

## 🔄 Workflow

```
SolidWorks Assembly Image
            │
            ▼
Dataset Preparation
            │
            ▼
YOLOv8 Training
            │
            ▼
Trained Model (finalbest.pt)
            │
            ▼
Object Detection
            │
            ▼
Component Counting
            │
            ▼
Annotated Output Image
```

---

## 📸 Sample Result

### Input

```
custom_test/image.png
```

### Output

The model generates an annotated image:

```
yolo_output.png
```

Example prediction:

| Component | Count |
|----------|------:|
| Bolt | 0 |
| Nut | 1 |
| Washer | 0 |
| Locating Pin | 0 |

---

## 💡 Future Improvements

- Real-time webcam detection
- ONNX/TensorRT deployment
- Integration with CAD inspection software
- Confidence threshold optimization
- Support for additional mechanical components

---

## 👩‍💻 Author

**Pooja Pandian**

B.Tech Artificial Intelligence & Data Science

Kumaraguru College of Technology

GitHub: https://github.com/poojapandian6

LinkedIn: *(Add your LinkedIn profile URL here)*

---

## 📜 License

This project is licensed under the MIT License.
