import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

# Load model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

# Load labels
with open("imagenet_labels.json") as f:
    labels = json.load(f)

# Preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open("image_for_testing/test_image.jpeg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)




\



# Run inference
with torch.no_grad():
    logits = model(input_tensor)

# Convert to probabilities
probabilities = torch.nn.functional.softmax(logits[0], dim=0)
top5 = torch.topk(probabilities, 5)

# Print explanation
print(f"── Raw output shape: {logits.shape}")
print(f"   (1 image, 1000 possible classes)\n")
print(f"── Highest raw logit: {logits[0].max().item():.4f}")
print(f"── Lowest raw logit:  {logits[0].min().item():.4f}")
print(f"── Probabilities sum: {probabilities.sum().item():.4f} (always 1.0)\n")

print("── Top 5 Predictions ───────────────────────────────")
for i in range(5):
    idx  = top5.indices[i].item()
    prob = top5.values[i].item() * 100
    bar  = "█" * int(prob / 2)
    print(f"  {i+1}. {labels[idx]:<35} {prob:5.2f}%  {bar}")