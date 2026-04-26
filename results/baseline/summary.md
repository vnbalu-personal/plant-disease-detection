# Binary Plant Disease Baseline

## Methodology
- Dataset: 20,638 unique PlantVillage images after deduplicating the archive's mirrored directory trees.
- Labels: folders containing `healthy` are mapped to `healthy`; every other original PlantVillage folder is mapped to `diseased`.
- Split: fixed 80/20 train-test partition with `random_state=42`, stratified by original source folder so each plant-condition subtype remains represented.
- Preprocessing: images are resized to 64x64 RGB PNG files, then converted to grayscale during feature extraction.
- Features and model: Histogram of Oriented Gradients (orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) followed by `StandardScaler` and `LinearSVC(class_weight="balanced")`.

## Evaluation Metrics
- Accuracy measures overall correctness.
- Balanced accuracy adjusts for the strong healthy vs diseased class imbalance.
- Precision, recall, and F1 are reported for the `diseased` class so missed diseased leaves are visible.
- ROC-AUC is computed from the SVM decision scores.
- The confusion matrix is included to show false positives and false negatives explicitly.

## Baseline Performance
- Accuracy: 86.02%
- Balanced accuracy: 82.73%
- Precision (`diseased`): 95.55%
- Recall (`diseased`): 87.51%
- F1 (`diseased`): 91.36%
- ROC-AUC: 91.75%
- Confusion matrix (`actual x predicted`, labels ordered as `[healthy, diseased]`): [[502, 142], [435, 3049]]

This baseline is intentionally simple and reproducible, but PlantVillage is a controlled laboratory-style dataset. Strong held-out performance here can still overestimate how well the same model will generalize to real field images with different lighting, backgrounds, and camera quality.