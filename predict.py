import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------
MODEL_PATH = os.path.join("saved_models", "resnet18_best.pth")
IMAGE_PATH = os.path.join("custom_test", "image.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------- MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- LOAD IMAGE ----------------
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ---------------- PREDICT ----------------
with torch.no_grad():
    prediction = model(input_tensor)

pred = prediction.cpu().numpy()[0]

result = {
    "bolt": max(0, round(pred[0])),
    "locatingpin": max(0, round(pred[1])),
    "nut": max(0, round(pred[2])),
    "washer": max(0, round(pred[3]))
}

print("\nPrediction for new image:")
print(result)

# ---------------- SHOW IMAGE ----------------
plt.imshow(image)
plt.axis("off")
plt.title("Custom Test Image")
plt.show()
