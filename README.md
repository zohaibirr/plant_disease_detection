Plant Disease Detection using Vision Transformer (ViT) and CNN Baseline
Overview
This project implements an image-based plant disease detection system using deep learning. The goal is to classify plant leaf images into multiple disease and healthy classes using the PlantVillage dataset. Two models are developed and compared:

A simple Convolutional Neural Network (CNN) baseline.
A Vision Transformer (ViT) model (vit_base_patch16_224) fine‑tuned from ImageNet weights using the timm library.
The pipeline covers dataset download from Kaggle, train/validation/test splitting, preprocessing, model training, evaluation, visualization, and single‑image inference. The work is motivated by the need for early, automated plant disease diagnosis to support SDG 2 (Zero Hunger) and SDG 12 (Responsible Consumption and Production).

1. Introduction
1.1 Background and Motivation
Plant diseases are a major cause of yield loss and economic risk. Diagnosis is usually performed visually by farmers or experts, which is:

Slow and subjective
Dependent on access to expertise
Prone to misdiagnosis and delayed treatment
The availability of large labeled datasets like PlantVillage and advances in computer vision (CNNs, Vision Transformers) enable automated tools that can diagnose diseases from leaf images captured on smartphones. Such tools can support farmers and extension workers by providing fast, consistent predictions.

1.2 Objectives
The main objectives of this project are:

Build a reproducible PyTorch pipeline for:
Downloading and splitting the PlantVillage dataset
Preprocessing and loading images
Training, evaluating, and visualizing models
Implement and train a simple CNN baseline for plant disease classification.
Fine‑tune a ViT model on the same dataset using transfer learning.
Evaluate both models using accuracy, precision, recall, F1‑score, confusion matrices, and ROC/PR curves.
Compare CNN and ViT performance and discuss trade‑offs for deployment.
Relate the solution to SDG 2 and SDG 12.
2. Dataset
2.1 Source
Dataset: PlantVillage (color images)
Source: Kaggle
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
The dataset contains color images of individual plant leaves labeled with crop–disease or healthy categories.

2.2 Structure
After downloading and unzipping via the Kaggle API, the raw data is placed under:

data/plantvillage_raw/
A splitter script (utils/split_dataset.py) then:

Locates the color/ subset under the raw folder.
For each class directory, shuffles and splits images into:
Train: 70%
Validation: 15%
Test: 15%
Copies images into:
text

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

Convert images to RGB (handled by ImageFolder)
Resize to a fixed size
Convert to tensors
Normalize with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]
Data augmentation:

RandomHorizontalFlip() on the training set
Different input sizes per model:

Baseline CNN
Input size: 128×128

ViT
Input size: 224×224 (required by vit_base_patch16_224)

3. Methodology
3.1 Models
3.1.1 Baseline CNN
Defined in model.py as SimpleCNN:

Feature extractor:

Conv2d(3 → 32, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(32 → 64, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(64 → 128, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
After three pooling operations, the spatial resolution is reduced from 128×128 to 16×16.

Classifier:

Flatten
Linear(128 × 16 × 16 → 256) → ReLU
Dropout(0.5)
Linear(256 → num_classes)
This architecture is intentionally lightweight to provide a clear baseline.

3.1.2 Vision Transformer (ViT)
Defined in model.py as build_vit:

Uses timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
Loads ImageNet-pretrained weights
Replaces the final classification head to output logits for the PlantVillage classes
Works on 224×224 input images
ViT splits the image into 16×16 patches, embeds them, and processes them with transformer encoder layers, using a special classification token for the final prediction.

3.2 Training Procedure
The main training logic is in train.py:

Common training loop _train_loop:

Loss: nn.CrossEntropyLoss
Optimizer: Adam
Tracks training loss and validation accuracy per epoch
train_baseline_cnn:

Uses SimpleCNN with image size 128
Typically: batch size 32, learning rate 1e-3, 3 epochs (prototype)
train_vit:

Uses ViT with image size 224
Typically: batch size 16, learning rate 3e-4, 3 epochs
Device is chosen automatically with get_device():

GPU if available, otherwise CPU.
4. Experimental Setup
4.1 Data Splits
From utils/split_dataset.py:

Train: 70% of images per class
Validation: 15% per class
Test: 15% per class
Splitting is done per class (stratified) to maintain class balance across splits.

4.2 Hyperparameters
Baseline CNN

Image size: 128×128
Optimizer: Adam (lr = 1e-3)
Batch size: 32
Epochs: 3
Loss: CrossEntropyLoss
Vision Transformer (ViT)

Image size: 224×224
Optimizer: Adam (lr = 3e-4)
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
Dataloaders: torchvision.datasets.ImageFolder, DataLoader with num_workers=0 (to avoid multiprocessing issues in Colab)
5. Results
Replace the XX.XX placeholders with your actual test metrics from
artifacts_baseline.pkl and artifacts_vit.pkl.

5.1 Quantitative Results
Baseline CNN (128×128)

Test Accuracy: XX.XX %
Weighted Precision: XX.XX %
Weighted Recall: XX.XX %
Weighted F1-score: XX.XX %
Vision Transformer (ViT, 224×224)

Test Accuracy: YY.YY %
Weighted Precision: YY.YY %
Weighted Recall: YY.YY %
Weighted F1-score: YY.YY %
5.1.1 Comparison Table
Model	Accuracy	Precision (weighted)	Recall (weighted)	F1-score (weighted)
Baseline CNN	XX.XX %	XX.XX %	XX.XX %	XX.XX %
ViT (ViT-B/16)	YY.YY %	YY.YY %	YY.YY %	YY.YY %
5.2 Visualizations
Generated by visualize_results.py in docs/figures/:

Training curves
baseline_cnn_loss_curve.png
baseline_cnn_accuracy_curve.png
vit_loss_curve.png
vit_accuracy_curve.png
Confusion matrices
baseline_cnn_confusion_matrix.png
vit_confusion_matrix.png
ROC and Precision–Recall (micro-averaged)
baseline_cnn_roc_curve_micro.png
baseline_cnn_precision_recall_curve_micro.png
vit_roc_curve_micro.png
vit_precision_recall_curve_micro.png
These show convergence behavior, class-wise performance, and overall discriminative ability.

6. Discussion
The CNN baseline achieves strong performance with low computational cost. It is straightforward to train and suitable for devices with limited resources.
The ViT model benefits from pretrained features and can capture more global context, which is helpful for subtle disease differences. However, it requires significantly more training time and GPU memory.
Depending on the final metrics, ViT may or may not significantly outperform the CNN. This highlights that simple models, when well implemented, can remain competitive on curated datasets.
Challenges:
Training ViT is relatively slow in Colab (≈ 20+ minutes per epoch).
Colab’s multiprocessing behavior required num_workers=0 for stable dataloaders.
Lessons learned:
A modular project structure (utils/, model.py, train.py, evaluate.py, inference.py) simplifies debugging and experimentation.
Transfer learning with ViT is powerful but must be balanced against resource constraints.
7. Installation & Usage
7.1 Requirements
Create a requirements.txt with:

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
7.2 Running in Google Colab (Recommended)
Set up project directories

Python

!rm -rf /content/plant_disease_vit

import os, sys
PROJECT_ROOT = "/content/plant_disease_vit"
DATA_DIR_RAW = os.path.join(PROJECT_ROOT, "data", "plantvillage_raw")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "plantvillage")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

for d in [PROJECT_ROOT, DATA_DIR_RAW, DATA_DIR, FIGURES_DIR, UTILS_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.append(PROJECT_ROOT)
Install dependencies

Python

!pip install timm kaggle -q
Configure Kaggle

Upload kaggle.json (Kaggle API token).
Move and set permissions:
Python

from google.colab import files
files.upload()  # select kaggle.json

import shutil, os
os.makedirs("/root/.kaggle", exist_ok=True)
shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)
Download and unzip dataset

Python

!kaggle datasets download -d abdallahalidev/plantvillage-dataset -p "{DATA_DIR_RAW}" -q
!unzip -oq "{DATA_DIR_RAW}/plantvillage-dataset.zip" -d "{DATA_DIR_RAW}"
Create project files

Run the %%writefile cells for:
utils/__init__.py, utils/helpers.py, utils/data_loader.py, utils/metrics.py, utils/split_dataset.py
model.py, train.py, evaluate.py, visualize_results.py, inference.py
Split dataset

Python

!python /content/plant_disease_vit/utils/split_dataset.py
Train models

Python

from train import train_baseline_cnn, train_vit
from utils.helpers import ensure_dir

ROOT = PROJECT_ROOT
DATA_ROOT = DATA_DIR
FIGURES = FIGURES_DIR
ensure_dir(FIGURES)

baseline_result = train_baseline_cnn(...)
vit_result = train_vit(...)
Evaluate & visualize

Use evaluate_on_test from evaluate.py.
Use functions from visualize_results.py to generate figures.
Inference on a single image

Python

from inference import predict_image
pred_class, conf = predict_image(...)
8. Project Structure
text

plant_disease_vit/
├── data/
│   ├── plantvillage_raw/
│   └── plantvillage/
│       ├── train/
│       ├── val/
│       └── test/
├── docs/
│   └── figures/
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── split_dataset.py
├── model.py
├── train.py
├── evaluate.py
├── visualize_results.py
├── inference.py
├── baseline_cnn.pth
├── vit_classifier.pth
├── training_history_baseline.pkl
├── training_history_vit.pkl
├── artifacts_baseline.pkl
└── artifacts_vit.pkl
9. Conclusion & Future Work
This project demonstrates a complete pipeline for plant disease detection from leaf images using a CNN baseline and a ViT model trained on the PlantVillage dataset. It shows how simple CNNs can reach strong performance, while ViTs can further leverage transfer learning at higher computational cost.

Future directions:

More extensive hyperparameter tuning and learning rate schedules.
Stronger data augmentation (color jitter, random rotations, Mixup/CutMix).
Model compression (distillation, pruning) to deploy ViT on edge devices.
Training with more realistic field images to improve real‑world generalization.
10. References
PlantVillage Dataset (Kaggle):
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
A. Dosovitskiy et al., “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale”, 2020.
timm library: https://github.com/huggingface/pytorch-image-models
PyTorch documentation: https://pytorch.org/docs/stable/
11. Acknowledgments
PlantVillage and Kaggle for providing the dataset.
The PyTorch and timm communities for core deep learning tools.
Google Colab for GPU resources used in training and experiments.
