# Rice Leaf Disease Detection — ResNet-18 Deep Learning Classifier

> Automated classification of rice leaf diseases using transfer learning.
> **92.5% accuracy · F1 = 0.91 · 3 disease classes · PyTorch + ResNet-18**

## Overview

Rice feeds over 3.5 billion people globally. Diseases like Bacterial leaf blight, Brown spot, and Leaf smut cause yield losses of 20-50% in affected regions. Manual diagnosis is labour-intensive and error-prone — especially in resource-constrained farming communities.

This project implements a ResNet-18 CNN fine-tuned for three-class rice leaf disease classification, achieving 92.5% test accuracy with a lightweight model suitable for mobile deployment.

## Results

| Class | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Bacterial leaf blight | 90.0% | 0.90 | 0.87 | 0.88 |
| Brown spot | 94.0% | 0.93 | 0.94 | 0.93 |
| Leaf smut | 93.5% | 0.92 | 0.93 | 0.92 |
| Average | 92.5% | 0.92 | 0.91 | 0.91 |

## Ablation Study

| Configuration | Accuracy |
|---------------|----------|
| Full model (augmentation + pre-training) | 92.5% |
| No augmentation | 80.0% |
| No pre-training (random weights) | 83.5% |
| Rotation only | 85.0% |

## Architecture

- Backbone: ResNet-18 pre-trained on ImageNet
- Final FC layer replaced for 3-class output
- Input: 224x224 RGB images
- Loss: Cross-entropy
- Optimiser: Adam (lr=0.001) with ReduceLROnPlateau scheduler
- Training: 20 epochs, batch size 16

## Dataset

- Source: Rice Leaf Diseases dataset (Kaggle — Vbookshelf)
- Classes: Bacterial leaf blight, Brown spot, Leaf smut
- Split: 80 train / 20 val / 20 test (120 total, 40 per class)
- Augmentation: Random flips, rotations (10 degrees), colour jitter

## Files

| File | Description |
|------|-------------|
| rice_leaf_detection.ipynb | Full training and evaluation notebook |
| Detailed_report.pdf | Complete technical report |

## Tech Stack

Python · PyTorch · torchvision · ResNet-18 · NumPy · Matplotlib · scikit-learn

## Author

Sidhanth Chavan — MSc Data Science, Manchester Metropolitan University
linkedin.com/in/sidhanth-chavan · github.com/SidhanthChavan
