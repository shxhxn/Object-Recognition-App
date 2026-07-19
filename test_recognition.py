import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import json
import os

# Load model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

# Download ImageNet labels (1000 class names)
labels_path = "imagenet_labels.json"
if not os.path.exists(labels_path):
    print("Downloading ImageNet labels...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
        labels_path
    )
    print("Labels downloaded.")

with open(labels_path) as f:
    labels = json.load(f)

image_path = "image_for_testing/my_personal_mouse.jpeg"

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)  # add batch dimension

# Run inference
with torch.no_grad():
    output = model(input_batch)

# Get top 5 predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5 = torch.topk(probabilities, 5)

print("\n── Top 5 Predictions ──────────────────────────")
for i in range(5):
    idx = top5.indices[i].item()
    prob = top5.values[i].item() * 100
    print(f"  {i+1}. {labels[idx]:<30} {prob:.2f}%")