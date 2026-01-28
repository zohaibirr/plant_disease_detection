# Plant Disease Detection using Vision Transformer (ViT)

## Abstract
This project applies Vision Transformers (ViT) to classify plant leaf diseases from images.
The model leverages self-attention mechanisms to capture global visual patterns, achieving
high accuracy in agricultural disease diagnosis.

## Domain
Agriculture

## Algorithm
Vision Transformer (Transformer-based Deep Learning Model)

## Dataset
Image-based plant disease dataset organized into train, validation, and test sets.
Preprocessing includes resizing, normalization, and augmentation.

## Methodology
A pretrained ViT-Base model is fine-tuned on the dataset. Cross-entropy loss and Adam
optimizer are used for training.

## Results
- Accuracy: >75%
- Metrics: Precision, Recall, F1-score
- Visualizations: Confusion Matrix, Learning Curves

## Installation
```bash
pip install -r requirements.txt
