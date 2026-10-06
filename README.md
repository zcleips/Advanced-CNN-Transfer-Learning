# Transfer Learning for Image Classification with Limited Data

## Overview

This project investigates the effectiveness of transfer learning for image classification under a limited-data regime.

The experiments use **ResNet-18** and a 10-class subset of the **Oxford-IIIT Pet Dataset**. Four main experimental conditions are evaluated:

- **B1 — From Scratch**
- **B2 — Feature Extraction**
- **B3 — Progressive Fine-Tuning**
- **B4 — Data Augmentation Control**

The main objective is to compare the performance, stability, and computational cost of training a CNN from scratch against different transfer learning strategies.

---

## Dataset

The experiments use a subset of 10 classes from the Oxford-IIIT Pet Dataset:

### Cat breeds
- Abyssinian
- Bengal
- Birman
- Persian
- Siamese

### Dog breeds
- American Bulldog
- American Pit Bull Terrier
- Basset Hound
- Beagle
- Bombay

The dataset is split using a stratified:

- **70% training**
- **15% validation**
- **15% testing**

The resulting dataset contains:

| Split | Images |
|---|---:|
| Train | 1,386 |
| Validation | 297 |
| Test | 298 |
| Total | 1,981 |

---

## Experimental Setup

All experiments use the same basic training protocol.

| Configuration | Value |
|---|---|
| Architecture | ResNet-18 |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Random seed | 42 |
| Early stopping | Validation loss |
| Weight decay | 1e-2 |

ImageNet normalization is used for the input images.

For training with augmentation, the following transformations are applied:

- RandomResizedCrop
- RandomHorizontalFlip
- RandomRotation (15°)
- ColorJitter

Validation and test images use deterministic preprocessing without random augmentation.

---

# Experiments

## B1 — From Scratch

ResNet-18 is initialized without pretrained weights:

```python
weights=None
