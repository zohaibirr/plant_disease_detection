import torch
from src.dataset import get_dataloaders
from src.model import build_vit
from src.train import train
from src.evaluate import save_results

DATA_DIR = "data"
EPOCHS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dl, val_dl, test_dl, classes = get_dataloaders(DATA_DIR)
model = build_vit(len(classes))

train(model, train_dl, val_dl, device, EPOCHS)
save_results(model, test_dl, device)
