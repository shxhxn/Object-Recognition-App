import torchvision.models as models

# Load MobileNetV2 with pretrained ImageNet weights
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Set to evaluation mode
# .eval() disables dropout and batchnorm training behavior
# Always call this when using a model for prediction, not training
model.eval()

print("Model loaded successfully!")
print(f"Model type: {type(model).__name__}")