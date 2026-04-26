# Plant Disease Detection Using CNNs

This project detects whether a plant leaf image is `healthy` or `diseased` using the PlantVillage dataset. It compares a classical machine-learning baseline against neural-network models:

- HOG features + LinearSVC baseline
- Custom CNN trained from scratch
- ResNet-18 transfer learning with a frozen backbone
- ResNet-18 transfer learning with the final block fine-tuned

The best model in the completed experiments was the fine-tuned ResNet-18.

## Dataset

Dataset: PlantVillage from Kaggle  
URL: <https://www.kaggle.com/datasets/emmarex/plantdisease>

The raw dataset and preprocessed images are not committed to this repository. They should stay local under `artifacts/`, which is ignored by Git.

## Results

Summary results are committed under `results/`.

| Model | Accuracy | Balanced Accuracy | Precision Diseased | Recall Diseased | F1 Diseased | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| HOG + LinearSVC baseline | 86.02% | 82.73% | 95.55% | 87.51% | 91.36% | 91.75% |
| Custom CNN | 98.79% | 97.70% | 99.28% | 99.28% | 99.28% | 99.80% |
| ResNet-18 frozen backbone | 96.92% | 96.60% | 99.27% | 97.07% | 98.16% | 99.35% |
| ResNet-18 fine-tuned last block | 99.73% | 99.65% | 99.91% | 99.77% | 99.84% | 100.00% |

See [results/model_comparison.md](results/model_comparison.md) for the full comparison and confusion matrices.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Preprocess

Download the Kaggle PlantVillage archive and point `--archive` to the local zip file.

```powershell
python plant-disease.py preprocess `
  --archive "C:\Users\ashwi\Downloads\archive (1).zip" `
  --output-dir ".\artifacts\preprocessed-224" `
  --test-size 0.2 `
  --val-size 0.1 `
  --random-state 42 `
  --image-size 224
```

This creates a train/validation/test split and a manifest at:

```text
artifacts/preprocessed-224/manifest.csv
```

## Train Models

Custom CNN:

```powershell
python plant-disease.py train-cnn `
  --manifest ".\artifacts\preprocessed-224\manifest.csv" `
  --output-dir ".\artifacts\cnn" `
  --epochs 8 `
  --batch-size 16
```

Frozen ResNet-18:

```powershell
python plant-disease.py train-resnet18 `
  --manifest ".\artifacts\preprocessed-224\manifest.csv" `
  --output-dir ".\artifacts\resnet18" `
  --epochs 5 `
  --batch-size 8 `
  --freeze-backbone
```

Fine-tuned ResNet-18:

```powershell
python plant-disease.py train-resnet18 `
  --manifest ".\artifacts\preprocessed-224\manifest.csv" `
  --output-dir ".\artifacts\resnet18-finetune" `
  --epochs 8 `
  --batch-size 8 `
  --freeze-backbone `
  --fine-tune-last-block
```

## Predict

```powershell
python plant-disease.py predict `
  --checkpoint ".\artifacts\resnet18-finetune\checkpoint.pt" `
  --image ".\example_leaf.jpg"
```

## Repository Notes

- `artifacts/` is ignored because it contains datasets, checkpoints, plots, and generated training outputs.
- `results/` is committed because it contains lightweight summaries, metrics, confusion matrices, and training histories for reporting.
- PlantVillage is a controlled dataset, so the high test accuracy may not fully represent performance on real field images.
