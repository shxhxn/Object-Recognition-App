# Object Recognition Prototype

A Python exploration of image classification with PyTorch and a pretrained MobileNetV2 model.

## What is included

- Load MobileNetV2 with ImageNet weights
- Prepare images with the matching torchvision transforms
- Run inference and inspect the top predictions
- Explore raw model outputs and class labels
- Start a minimal Flask static-file server

## Run the experiments

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python test_recognition.py
```

The scripts may download pretrained model weights and sample resources on first use, so an internet connection can be required.

## Prototype status

This is a learning prototype, not a finished application. `serve.py` expects a `webapp/` interface that is not currently included in the repository. The inference scripts are the usable part of the project today.

## Tech stack

Python · PyTorch · torchvision · Pillow · Flask
