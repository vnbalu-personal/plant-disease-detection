# Plant Disease Detection Results

## Dataset

- Source: PlantVillage Kaggle archive.
- Task: binary classification, `healthy` vs `diseased`.
- Total images: 20,638.
- Neural-network split: 14,446 train, 2,064 validation, 4,128 test.
- Class counts: 17,417 diseased, 3,221 healthy.

## Model Comparison

| Model | Accuracy | Balanced Accuracy | Precision Diseased | Recall Diseased | F1 Diseased | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| HOG + LinearSVC baseline | 86.02% | 82.73% | 95.55% | 87.51% | 91.36% | 91.75% |
| Custom CNN, trained from scratch | 98.79% | 97.70% | 99.28% | 99.28% | 99.28% | 99.80% |
| ResNet-18 frozen ImageNet backbone, reference only | 96.92% | 96.60% | 99.27% | 97.07% | 98.16% | 99.35% |
| ResNet-18 fine-tuned ImageNet backbone, reference only | 99.73% | 99.65% | 99.91% | 99.77% | 99.84% | 100.00% |

## Confusion Matrices

Labels are ordered as `[healthy, diseased]`, with rows as actual labels and columns as predicted labels.

```text
HOG + LinearSVC baseline:
[[502, 142],
 [435, 3049]]

Custom CNN, trained from scratch:
[[619, 25],
 [25, 3459]]

ResNet-18 frozen ImageNet backbone, reference only:
[[619, 25],
 [102, 3382]]

ResNet-18 fine-tuned ImageNet backbone, reference only:
[[641, 3],
 [8, 3476]]
```

## Conclusion

The custom CNN greatly improved over the classical HOG + LinearSVC baseline and is the selected competition model because its architecture is defined in this repository and trained from scratch. The ImageNet-pretrained ResNet-18 transfer-learning runs are retained as reference experiments only, not as the submitted model.

The ResNet-18 fine-tuning run had the highest measured test metrics, but it uses external pretrained ImageNet weights and a standard torchvision architecture. Under project rules that emphasize designing the neural network for the competition, those results should not be treated as the final submission result. All results should still be interpreted carefully because PlantVillage is a controlled image dataset and may overestimate performance on real field images.
