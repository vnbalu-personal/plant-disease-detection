# Quickstart

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## Preprocess PlantVillage

```powershell
python plant-disease.py preprocess `
  --archive "C:\Users\ashwi\Downloads\archive (1).zip" `
  --output-dir ".\artifacts\preprocessed" `
  --test-size 0.2 `
  --random-state 42 `
  --image-size 64
```

This command deduplicates the mirrored archive trees, maps classes to `healthy` or `diseased`, resizes images to `64x64`, and writes:

- `artifacts/preprocessed/manifest.csv`
- `artifacts/preprocessed/preprocess_summary.json`
- `artifacts/preprocessed/train/...`
- `artifacts/preprocessed/test/...`

## Train The Baseline

```powershell
python plant-disease.py train-baseline `
  --manifest ".\artifacts\preprocessed\manifest.csv" `
  --output-dir ".\artifacts\baseline"
```

This command trains a `HOG + LinearSVC` classifier and writes:

- `artifacts/baseline/metrics.json`
- `artifacts/baseline/confusion_matrix.csv`
- `artifacts/baseline/summary.md`
