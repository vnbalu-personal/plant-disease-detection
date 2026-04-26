# Binary Plant Disease ResNet-18 Transfer Learning

## Methodology
- Dataset: 20,638 PlantVillage images split into 14,446 train, 2,064 validation, and 4,128 test images.
- Labels: folders containing `healthy` are mapped to `healthy`; every other original PlantVillage folder is mapped to `diseased`.
- Preprocessing: images are resized to 224x224 RGB PNG files and normalized before training.
- Model: ResNet-18 Transfer Learning with a two-class output head.
- Training: AdamW optimizer, weighted cross-entropy loss, batch size 8, best checkpoint selected at epoch 8.

## Test Performance
- Accuracy: 99.73%
- Balanced accuracy: 99.65%
- Precision (`diseased`): 99.91%
- Recall (`diseased`): 99.77%
- F1 (`diseased`): 99.84%
- ROC-AUC: 100.00%
- Confusion matrix (`actual x predicted`, labels ordered as `[healthy, diseased]`): [[641, 3], [8, 3476]]

This neural-network result should be compared with the HOG + LinearSVC baseline, while remembering that PlantVillage images are controlled lab-style images and may overestimate real field performance.