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
| Custom CNN | 98.79% | 97.70% | 99.28% | 99.28% | 99.28% | 99.80% |
| ResNet-18 frozen backbone | 96.92% | 96.60% | 99.27% | 97.07% | 98.16% | 99.35% |
| ResNet-18 fine-tuned last block | 99.73% | 99.65% | 99.91% | 99.77% | 99.84% | 100.00% |

## Confusion Matrices

Labels are ordered as `[healthy, diseased]`, with rows as actual labels and columns as predicted labels.

```text
HOG + LinearSVC baseline:
[[502, 142],
 [435, 3049]]

Custom CNN:
[[619, 25],
 [25, 3459]]

ResNet-18 frozen backbone:
[[619, 25],
 [102, 3382]]

ResNet-18 fine-tuned last block:
[[641, 3],
 [8, 3476]]
```

## Conclusion

The custom CNN greatly improved over the classical HOG + LinearSVC baseline. The first ResNet-18 transfer-learning run, with the backbone frozen, improved over the baseline but did not beat the custom CNN. After unfreezing and fine-tuning the final ResNet block, ResNet-18 achieved the best overall test performance.

The final selected model is the fine-tuned ResNet-18 because it had the highest accuracy, balanced accuracy, diseased-class F1, and the fewest total test errors. The result should still be interpreted carefully because PlantVillage is a controlled image dataset and may overestimate performance on real field images.
