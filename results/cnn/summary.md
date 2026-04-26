# Binary Plant Disease Custom CNN

## Methodology
- Dataset: 20,638 PlantVillage images split into 14,446 train, 2,064 validation, and 4,128 test images.
- Labels: folders containing `healthy` are mapped to `healthy`; every other original PlantVillage folder is mapped to `diseased`.
- Preprocessing: images are resized to 224x224 RGB PNG files and normalized before training.
- Model: Custom CNN with a two-class output head.
- Training: Adam optimizer, weighted cross-entropy loss, batch size 16, best checkpoint selected at epoch 7.

## Test Performance
- Accuracy: 98.79%
- Balanced accuracy: 97.70%
- Precision (`diseased`): 99.28%
- Recall (`diseased`): 99.28%
- F1 (`diseased`): 99.28%
- ROC-AUC: 99.80%
- Confusion matrix (`actual x predicted`, labels ordered as `[healthy, diseased]`): [[619, 25], [25, 3459]]

This neural-network result should be compared with the HOG + LinearSVC baseline, while remembering that PlantVillage images are controlled lab-style images and may overestimate real field performance.