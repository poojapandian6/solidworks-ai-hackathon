# 🔩 AI-Based Fastener Detection for SolidWorks Assemblies using YOLOv8

An AI-powered computer vision project that detects and classifies mechanical fasteners in SolidWorks assembly images using the YOLOv8 object detection model. The system identifies components such as **bolts, nuts, washers, and locating pins**, making it useful for automated inspection and manufacturing workflows.

---

## 📌 Project Overview

Manual inspection of mechanical assemblies is time-consuming and prone to human error. This project automates the detection of fasteners from SolidWorks assembly images using a deep learning-based object detection pipeline.

The model was trained on a custom dataset and performs inference on new assembly images, generating predictions along with annotated output images.

---

## ✨ Features

- Detects multiple fastener types
- YOLOv8-based object detection
- Custom dataset preprocessing
- Automatic feature extraction
- Prediction on unseen images
- Generates annotated output images
- Lightweight inference pipeline

---

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy
- Pandas
- Scikit-learn

---

## 📂 Project Structure

```
solidworks-ai-hackathon/
│
├── saved_models/
│   ├── finalbest.pt
│   └── yolo_corrector.pkl
│
├── custom_test/
│   └── image.png
│
├── build_yolo_features.py
├── clean_yolo_features.py
├── prepare_yolo_data.py
├── predict.py
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
git clone https://github.com/YOUR_USERNAME/solidworks-ai-hackathon.git

cd solidworks-ai-hackathon
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Prediction

Place the test image inside the `custom_test` folder.

Run:

```bash
python predict_yolo.py
```

The model predicts all detected fasteners and saves an annotated output image.

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

## 🧠 Workflow

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
 Prediction
            │
            ▼
 Detected Components
            │
            ▼
 Annotated Output Image
```

---

## 📸 Sample Output

Input image:

```
custom_test/image.png
```

Output image:

```
yolo_output.png
```

(Add screenshots here after uploading.)

---

## 📈 Future Improvements

- Real-time webcam detection
- Confidence threshold tuning
- ONNX/TensorRT deployment
- Integration with CAD inspection systems
- Web application for inference

---

## 👩‍💻 Author

**Pooja Pandian**

B.Tech Artificial Intelligence & Data Science

Kumaraguru College of Technology

GitHub: https://github.com/poojapandian6

## 📜 License

This project is released under the MIT License.
