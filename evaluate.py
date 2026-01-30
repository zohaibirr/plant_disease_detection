import os
import pickle

from utils.helpers import get_device, load_checkpoint
from utils.data_loader import create_dataloaders
from utils.metrics import evaluate_model
from model import build_vit, SimpleCNN


def evaluate_on_test(
    model_type: str,
    data_root: str,
    weights_path: str,
    image_size: int = 128,
    batch_size: int = 32,
    artifacts_path: str = "artifacts.pkl",
):
    device = get_device()

    # ViT model expects 224x224
    if model_type == "vit":
        image_size = 224

    _, _, test_loader, class_names = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,          # 🔴 force no multiprocessing
    )

    if model_type == "vit":
        model = build_vit(num_classes=len(class_names))
    elif model_type == "baseline_cnn":
        model = SimpleCNN(num_classes=len(class_names), image_size=image_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    load_checkpoint(model, weights_path, map_location=device)
    model.to(device)

    metrics = evaluate_model(model, test_loader, device)
    acc = metrics["accuracy"]
    print(f"[{model_type}] Test Accuracy: {acc*100:.2f}%")

    artifacts = {
        "model_type": model_type,
        "class_names": class_names,
        "metrics": metrics,
    }

    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)

    return artifacts
