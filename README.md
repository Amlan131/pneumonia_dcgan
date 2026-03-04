# DCGAN for Chest X-Ray Augmentation — PneumoniaMNIST

Training a DCGAN from scratch in PyTorch to generate synthetic chest X-rays 
and fix class imbalance in a pneumonia detection classifier.

## Problem
PneumoniaMNIST has a 3.5:1 class imbalance (Pneumonia vs Normal).  
A classifier trained naively defaults toward pneumonia — not because it 
learned pathology, but because it learned prevalence.

## Approach
1. Train a DCGAN on the minority class (Normal) only
2. Filter synthetic images using discriminator confidence score
3. Fill 50% of the class gap with synthetic images
4. Calibrate the decision threshold (0.5 → 0.35) to correct probability bias

## Results

| Metric        | Original | GAN-Augmented |
|---------------|----------|---------------|
| Accuracy      | 86.86%   | 86.38%        |
| Macro F1      | 0.8495   | 0.8421        |
| AUC-ROC       | 0.9419   | 0.9451        |
| Normal Recall | 0.57     | **0.66**      |

Accuracy barely moved. That was the point.  
In skewed medical datasets, real progress shows up in minority-class recall — not accuracy.

## Setup
Open pneumonia_dcgan.ipynb and run all cells top to bottom.
Dataset downloads automatically via the medmnist package.

Environment
Python 3.12

PyTorch 2.x

CUDA (optional, falls back to CPU)

```bash
pip install medmnist scikit-learn torch torchvision
