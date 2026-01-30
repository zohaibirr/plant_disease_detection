import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from utils.helpers import set_seed, get_device, save_checkpoint
from utils.data_loader import create_dataloaders
from utils.metrics import evaluate_model
from model import build_vit, SimpleCNN


def _train_loop(model, train_loader, val_loader, device, epochs, lr) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    history["best_val_acc"] = best_val_acc
    return history


def train_vit(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 3e-4,
    seed: int = 42,
    weights_path: str = "vit_classifier.pth",
) -> Dict[str, Any]:

    set_seed(seed)
    device = get_device()

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,          # 🔴 force no multiprocessing
    )

    model = build_vit(num_classes=len(class_names))
    model.to(device)

    history = _train_loop(model, train_loader, val_loader, device, epochs, lr)
    save_checkpoint(model, weights_path)

    return {
        "history": history,
        "class_names": class_names,
        "test_loader": test_loader,
        "device": device,
    }


def train_baseline_cnn(
    data_root: str,
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 1e-3,
    seed: int = 42,
    weights_path: str = "baseline_cnn.pth",
) -> Dict[str, Any]:

    set_seed(seed)
    device = get_device()

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,          # 🔴 force no multiprocessing
    )

    model = SimpleCNN(num_classes=len(class_names), image_size=image_size)
    model.to(device)

    history = _train_loop(model, train_loader, val_loader, device, epochs, lr)
    save_checkpoint(model, weights_path)

    return {
        "history": history,
        "class_names": class_names,
        "test_loader": test_loader,
        "device": device,
    }
