# Quickstart

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## Preprocess PlantVillage For Neural Networks

```powershell
python plant-disease.py preprocess `
  --archive "C:\Users\ashwi\Downloads\archive (1).zip" `
  --output-dir ".\artifacts\preprocessed-224" `
  --test-size 0.2 `
  --val-size 0.1 `
  --random-state 42 `
  --image-size 224
```

This command deduplicates the mirrored archive trees, maps classes to `healthy` or `diseased`, resizes images to `224x224`, and writes:

- `artifacts/preprocessed-224/manifest.csv`
- `artifacts/preprocessed-224/preprocess_summary.json`
- `artifacts/preprocessed-224/train/...`
- `artifacts/preprocessed-224/val/...`
- `artifacts/preprocessed-224/test/...`

## Train The Custom CNN

```powershell
python plant-disease.py train-cnn `
  --manifest ".\artifacts\preprocessed-224\manifest.csv" `
  --output-dir ".\artifacts\cnn" `
  --epochs 8 `
  --batch-size 16
```

## Train ResNet-18 Transfer Learning

```powershell
python plant-disease.py train-resnet18 `
  --manifest ".\artifacts\preprocessed-224\manifest.csv" `
  --output-dir ".\artifacts\resnet18" `
  --epochs 5 `
  --batch-size 8 `
  --freeze-backbone
```

Use `--fine-tune-last-block` to unfreeze the final ResNet block. If pretrained weights are not available offline, use `--no-pretrained` only for smoke testing.

Each neural training command writes:

- `checkpoint.pt`
- `metrics.json`
- `confusion_matrix.csv`
- `training_history.csv`
- `summary.md`
- `loss_accuracy_curve.png`

## Predict One Image

```powershell
python plant-disease.py predict `
  --checkpoint ".\artifacts\resnet18\checkpoint.pt" `
  --image ".\example_leaf.jpg"
```

## Optional Classical Baseline

The original HOG + LinearSVC baseline is still available for comparison:

```powershell
python plant-disease.py train-baseline `
  --manifest ".\artifacts\preprocessed-224\manifest.csv" `
  --output-dir ".\artifacts\baseline"
```
