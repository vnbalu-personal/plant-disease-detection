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

## Run

Download the PlantVillage dataset from Kaggle:

<https://www.kaggle.com/datasets/emmarex/plantdisease>

Then preprocess it:

```powershell
python plant-disease.py preprocess `
  --archive ".\data\plantvillage.zip" `
  --output-dir ".\artifacts\preprocessed-224" `
  --test-size 0.2 `
  --val-size 0.1 `
  --random-state 42 `
  --image-size 224
```

Train the custom CNN:

```powershell
python plant-disease.py train-cnn `
  --manifest ".\artifacts\preprocessed-224\manifest.csv" `
  --output-dir ".\artifacts\cnn" `
  --epochs 8 `
  --batch-size 16
```

Train the fine-tuned ResNet-18:

```powershell
python plant-disease.py train-resnet18 `
  --manifest ".\artifacts\preprocessed-224\manifest.csv" `
  --output-dir ".\artifacts\resnet18-finetune" `
  --epochs 8 `
  --batch-size 8 `
  --freeze-backbone `
  --fine-tune-last-block
```

## Notes

- `artifacts/` is ignored because it contains generated datasets, checkpoints, and plots.
- `results/` is committed because it contains small files needed for reporting.
- The dataset images are controlled lab-style images, so real-world field performance may be lower.
