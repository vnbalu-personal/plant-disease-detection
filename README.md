# Plant Disease Detection Using CNNs

This project uses image classification to predict whether a plant leaf is `healthy` or `diseased`. The dataset is PlantVillage from Kaggle, and the task is treated as a binary classification problem.

## Models

The project compares four approaches:

- HOG features with a LinearSVC classifier
- Custom CNN trained from scratch
- ResNet-18 with a frozen pretrained backbone
- ResNet-18 with the final block fine-tuned

## Results

| Model | Accuracy | Balanced Accuracy | F1 Diseased |
|---|---:|---:|---:|
| HOG + LinearSVC baseline | 86.02% | 82.73% | 91.36% |
| Custom CNN | 98.79% | 97.70% | 99.28% |
| ResNet-18 frozen backbone | 96.92% | 96.60% | 98.16% |
| ResNet-18 fine-tuned last block | 99.73% | 99.65% | 99.84% |

The best model was the fine-tuned ResNet-18. Full metrics, confusion matrices, and training histories are in [results/](results/).

## Setup

```powershell
python -m pip install -r requirements.txt
```
