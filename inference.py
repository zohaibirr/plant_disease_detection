import os
import pickle

import torch
from torchvision import transforms
from PIL import Image

from utils.helpers import get_device, load_checkpoint
from model import SimpleCNN, build_vit


def _get_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])


def _load_image(image_path: str, image_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    transform = _get_transform(image_size)
    x = transform(img).unsqueeze(0)  # (1, C, H, W)
    return x


def _load_model(
    model_type: str,
    num_classes: int,
    weights_path: str,
    image_size: int,
):
    device = get_device()

    if model_type == "baseline_cnn":
        model = SimpleCNN(num_classes=num_classes, image_size=image_size)
    elif model_type == "vit":
        # ViT expects 224x224, image_size is ignored inside model
        model = build_vit(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    load_checkpoint(model, weights_path, map_location=device)
    model.to(device)
    model.eval()

    return model, device


def predict_image(
    image_path: str,
    model_type: str,
    weights_path: str,
    artifacts_path: str,
):
    """
    Predict the class of a single leaf image.

    Args:
        image_path: Path to the input image.
        model_type: "baseline_cnn" or "vit".
        weights_path: Path to the trained weights (.pth).
        artifacts_path: Path to artifacts_baseline.pkl or artifacts_vit.pkl
                        (must contain 'class_names').

    Returns:
        pred_class (str), confidence (float in [0, 1])
    """
    # Load class names from artifacts file
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    class_names = artifacts["class_names"]

    # Set image size depending on model type
    if model_type == "baseline_cnn":
        image_size = 128
    elif model_type == "vit":
        image_size = 224
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load model and device
    model, device = _load_model(
        model_type=model_type,
        num_classes=len(class_names),
        weights_path=weights_path,
        image_size=image_size,
    )

    # Load and preprocess image
    x = _load_image(image_path, image_size).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_class = class_names[pred_idx]
    return pred_class, confidence
