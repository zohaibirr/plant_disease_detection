import json
from src.utils import evaluate_model

def save_results(model, test_loader, device):
    acc, report = evaluate_model(model, test_loader, device)

    with open("results/metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Test Accuracy: {acc*100:.2f}%")
