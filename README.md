# Plant Disease Detection Using CNNs

This project uses image classification to predict whether a plant leaf is `healthy` or `diseased`. The dataset is PlantVillage from Kaggle, and the task is treated as a binary classification problem.

## Models

The project includes one competition model plus baseline/reference experiments:

- HOG features with a LinearSVC classifier
- Custom CNN trained from scratch for the competition submission
- ImageNet-pretrained ResNet-18 transfer-learning runs as reference experiments only

## Results

| Model | Accuracy | Balanced Accuracy | F1 Diseased |
|---|---:|---:|---:|
| HOG + LinearSVC baseline | 86.02% | 82.73% | 91.36% |
| Custom CNN, trained from scratch | 98.79% | 97.70% | 99.28% |
| ResNet-18 frozen ImageNet backbone, reference only | 96.92% | 96.60% | 98.16% |
| ResNet-18 fine-tuned ImageNet backbone, reference only | 99.73% | 99.65% | 99.84% |

The selected competition model is the custom CNN because it is the neural network designed and trained in this repository from scratch. The ImageNet-pretrained ResNet-18 runs are kept only as transfer-learning reference experiments and are not the submission model. Full metrics, confusion matrices, and training histories are in [results/](results/).

## Setup

```powershell
python -m pip install -r requirements.txt
```
