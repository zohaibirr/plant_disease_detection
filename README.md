Markdown

# Plant Disease Detection using Vision Transformer (ViT) and CNN Baseline

## Overview

This project implements an image-based plant disease detection system using deep learning. The goal is to classify plant leaf images into multiple disease and healthy classes using the PlantVillage dataset. Two models are developed and compared:

- A **simple Convolutional Neural Network (CNN)** baseline.
- A **Vision Transformer (ViT)** model (`vit_base_patch16_224`) fine-tuned using pretrained ImageNet weights via the `timm` library.

The full pipeline covers dataset download from Kaggle, train/validation/test splitting, preprocessing, model training, evaluation, visualization, and single-image inference. The work is motivated by the need for early, automated plant disease diagnosis to support SDG 2 (Zero Hunger) and SDG 12 (Responsible Consumption and Production).

---

## 1. Introduction

### 1.1 Background and Motivation

Plant diseases are a major cause of crop yield loss and economic risk in agriculture. Diagnosis is often done visually by farmers or extension workers, which is:

- Slow and subjective.
- Dependent on access to experts.
- Prone to misdiagnosis and delayed treatment.

With the availability of large curated datasets like **PlantVillage** and the advances in computer vision (CNNs, Vision Transformers), there is an opportunity to build automated tools that can diagnose diseases from leaf images captured by smartphones or simple cameras.

This project explores such a solution by building and comparing:

- A lightweight CNN model.
- A more advanced Vision Transformer (ViT) model.

### 1.2 Objectives

1. Build a reproducible pipeline in PyTorch for:
   - Downloading and splitting the PlantVillage dataset.
   - Preprocessing and loading images.
   - Training, evaluating, and visualizing models.
2. Implement and train a **simple CNN** baseline for plant disease classification.
3. Fine-tune a **ViT** model on the same dataset using transfer learning.
4. Evaluate models using accuracy, precision, recall, F1-score, confusion matrices, and ROC/PR curves.
5. Compare CNN and ViT performance and discuss trade-offs for real-world deployment.
6. Discuss how this system contributes to **SDG 2** and **SDG 12**.

---

## 2. Dataset

### 2.1 Source

- **Name:** PlantVillage Dataset (color images)
- **Source:** Kaggle  
  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset  

The dataset contains color images of individual plant leaves labeled by crop and disease (or healthy).

### 2.2 Structure

After downloading and unzipping via Kaggle API, the raw data is placed in:

- `data/plantvillage_raw/`

A robust splitter script (`utils/split_dataset.py`) then:

- Looks for the `color` subset under the raw directory.
- For each class folder, shuffles and splits images into:
  - **Train:** 70%
  - **Validation:** 15%
  - **Test:** 15%
- Copies images into:

```text
data/plantvillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
Each subfolder under train/, val/, test/ corresponds to one specific crop–disease or healthy class.

2.3 Preprocessing
Implemented in utils/data_loader.py:

Common steps:
Convert images to RGB.
Resize to a fixed size.
Convert to tensors.
Normalize with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5].
Data augmentation:
RandomHorizontalFlip() on the training set.
Different input sizes per model:

Baseline CNN:
Input size: 128×128
ViT:
Input size: 224×224 (required by vit_base_patch16_224)
3. Methodology
3.1 Models
3.1.1 Baseline CNN
Defined in model.py as SimpleCNN:

Feature extractor:
Conv2d(3 → 32, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(32 → 64, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(64 → 128, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
Classifier:
Flatten
Linear(128 × (image_size/8) × (image_size/8) → 256) → ReLU
Dropout(0.5)
Linear(256 → num_classes)
This architecture is intentionally simple and light, providing a clear baseline.

3.1.2 Vision Transformer (ViT)
Defined in model.py as build_vit:

Uses timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes).
Loads ImageNet-pretrained weights.
Replaces the final classification head to output logits over the PlantVillage classes.
Works on 224×224 input images.
3.2 Training Procedures
Both training loops are implemented in train.py:

Shared training loop _train_loop:
Loss: nn.CrossEntropyLoss
Optimizer: Adam
Tracks training loss and validation accuracy per epoch.
train_baseline_cnn:
Uses SimpleCNN with image size 128.
train_vit:
Uses ViT with image size 224.
Device automatically chosen via get_device():
GPU if available, otherwise CPU.
4. Experimental Setup
4.1 Data Splits
From utils/split_dataset.py:

Train: 70% per class
Validation: 15% per class
Test: 15% per class
Splitting is done per class (stratified) to maintain class balance across splits.

4.2 Hyperparameters
Baseline CNN

Image size: 128×128
Optimizer: Adam
Learning rate: 1e-3
Batch size: 32
Epochs: 3 (prototype; extendable)
Loss: CrossEntropyLoss
ViT

Image size: 224×224
Optimizer: Adam
Learning rate: 3e-4
Batch size: 16
Epochs: 3
Loss: CrossEntropyLoss
4.3 Environment
Platform: Google Colab (GPU runtime)
Language: Python 3
Libraries:
torch, torchvision
timm
scikit-learn
numpy, pandas
matplotlib, seaborn
kaggle (API)
Data loading: torchvision.datasets.ImageFolder, torch.utils.data.DataLoader with num_workers=0 (avoids multiprocessing issues in Colab).
5. Results
Replace TODO values with your final test metrics from artifacts_baseline.pkl and artifacts_vit.pkl.

5.1 Quantitative Results
Baseline CNN (128×128)

Test Accuracy: TODO %
Weighted Precision: TODO %
Weighted Recall: TODO %
Weighted F1-score: TODO %
Vision Transformer (ViT, 224×224)

Test Accuracy: TODO %
Weighted Precision: TODO %
Weighted Recall: TODO %
Weighted F1-score: TODO %
Comparison Table
Model	Accuracy	Precision (weighted)	Recall (weighted)	F1-score (weighted)
Baseline CNN	TODO %	TODO %	TODO %	TODO %
ViT (ViT-B/16)	TODO %	TODO %	TODO %	TODO %
5.2 Visualizations
Generated by visualize_results.py under docs/figures/:

Training curves
baseline_cnn_loss_curve.png
baseline_cnn_accuracy_curve.png
vit_loss_curve.png
vit_accuracy_curve.png
Confusion matrices
baseline_cnn_confusion_matrix.png
vit_confusion_matrix.png
ROC and Precision–Recall curves (micro-averaged)
baseline_cnn_roc_curve_micro.png
baseline_cnn_precision_recall_curve_micro.png
vit_roc_curve_micro.png
vit_precision_recall_curve_micro.png
These plots illustrate:

Convergence behavior (loss/accuracy curves).
Where each model tends to misclassify (confusion matrix).
Overall discriminative performance (ROC/PR curves).
6. Discussion
The CNN baseline provides strong performance with relatively low computational cost. It is easy to train and is suitable for deployment on lower-power devices.
The ViT model leverages pretrained features from a large-scale dataset and can capture more global context, which is useful for subtle visual differences between diseases.
In practice, CNN and ViT may achieve similar accuracies on this task; the exact advantage of ViT depends on:
Number of training epochs.
Data augmentations.
Hyperparameter tuning (learning rate, weight decay).
Challenges:
Training time for ViT is significantly longer than CNN (observed ~22 minutes per epoch vs ~1–2 minutes for CNN in Colab).
Colab’s environment can produce multiprocessing warnings, requiring num_workers=0.
Lessons learned:
A well-tuned simple model can be surprisingly strong; more complex models do not always outperform without sufficient tuning.
Organizing the project into reusable modules (utils/, train.py, evaluate.py, inference.py) makes iteration and debugging much easier.
7. Installation & Usage
7.1 Requirements
You can list these in requirements.txt:

txt

torch
torchvision
timm
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
kaggle
7.2 Steps (Google Colab)
Clone or create project structure under /content/plant_disease_vit.

Install dependencies:

Bash

!pip install timm kaggle -q
Configure Kaggle API:

Upload kaggle.json (Kaggle API token) in Colab.
Move it to /root/.kaggle/kaggle.json and set permissions.
Download dataset:

Bash

!kaggle datasets download -d abdallahalidev/plantvillage-dataset -p "{DATA_DIR_RAW}" -q
!unzip -oq "{DATA_DIR_RAW}/plantvillage-dataset.zip" -d "{DATA_DIR_RAW}"
Run %%writefile cells for all modules (utils/*.py, model.py, train.py, evaluate.py, visualize_results.py, inference.py).

Split dataset:

Bash

!python /content/plant_disease_vit/utils/split_dataset.py
Train models:

Python

from train import train_baseline_cnn, train_vit
from utils.helpers import ensure_dir

ROOT = PROJECT_ROOT
DATA_ROOT = DATA_DIR
FIGURES = FIGURES_DIR
ensure_dir(FIGURES)

baseline_result = train_baseline_cnn(...)
vit_result = train_vit(...)
Evaluate and generate plots:

Python

from evaluate import evaluate_on_test
from visualize_results import *

artifacts_baseline = evaluate_on_test(...)
artifacts_vit = evaluate_on_test(...)
# then call plot_training_curves, plot_confusion_matrix, plot_roc_pr_curves
Run inference on a leaf image:

Python

from inference import predict_image
pred_class, conf = predict_image(...)
8. Project Structure
text

plant_disease_vit/
├── data/
│   ├── plantvillage_raw/        # raw Kaggle download
│   └── plantvillage/            # train/val/test splits (ImageFolder)
├── docs/
│   └── figures/                 # PNGs: loss, accuracy, CM, ROC/PR
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── split_dataset.py
├── model.py                     # SimpleCNN baseline + ViT
├── train.py                     # training loops
├── evaluate.py                  # test evaluation + artifacts
├── visualize_results.py         # plotting utilities
├── inference.py                 # single-image prediction
├── baseline_cnn.pth             # trained CNN weights
├── vit_classifier.pth           # trained ViT weights
├── training_history_baseline.pkl
├── training_history_vit.pkl
├── artifacts_baseline.pkl
└── artifacts_vit.pkl
9. Conclusion & Future Work
This project demonstrates a complete deep learning pipeline for plant disease detection from leaf images, comparing a simple CNN baseline with a ViT model on the PlantVillage dataset. It shows that:

A well-designed CNN can provide strong performance with lower computational cost.
ViT can offer competitive or superior performance when adequately fine‑tuned, at the cost of longer training times.
Future improvements:

More extensive hyperparameter tuning (learning rates, schedulers, weight decay).
Stronger data augmentation (color jitter, random rotations, CutMix/Mixup).
Model distillation or pruning to compress ViT for edge deployment.
Training with more field-like images (non‑lab backgrounds) for better real‑world generalization.
10. References
PlantVillage Dataset (Kaggle):
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”, 2020.
timm library:
https://github.com/huggingface/pytorch-image-models
PyTorch documentation:
https://pytorch.org/docs/stable/index.html
11. Acknowledgments
Kaggle and the PlantVillage project for providing the dataset.
The authors and maintainers of PyTorch and timm for the core deep learning tools.
Google Colab for providing free GPU resources for training and experimentation.





