from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_SIZE = 224
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42
DEFAULT_CNN_EPOCHS = 8
DEFAULT_RESNET_EPOCHS = 5
DEFAULT_CNN_BATCH_SIZE = 16
DEFAULT_RESNET_BATCH_SIZE = 8
DEFAULT_CNN_LR = 1e-3
DEFAULT_RESNET_LR = 1e-4
POSITIVE_LABEL = "diseased"
NEGATIVE_LABEL = "healthy"
LABEL_TO_INDEX = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
MANIFEST_NAME = "manifest.csv"
PREPROCESS_SUMMARY_NAME = "preprocess_summary.json"
METRICS_NAME = "metrics.json"
CONFUSION_MATRIX_NAME = "confusion_matrix.csv"
SUMMARY_NAME = "summary.md"
CHECKPOINT_NAME = "checkpoint.pt"
HISTORY_NAME = "training_history.csv"
PLOT_NAME = "loss_accuracy_curve.png"


@dataclass(frozen=True)
class ImageRecord:
    archive_member: str
    normalized_relpath: str
    source_class: str
    binary_label: str


class ManifestImageDataset:
    def __init__(self, rows: list[dict[str, str]], dataset_root: Path, transform: object) -> None:
        self.rows = rows
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.dataset_root / row["stored_relpath"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, LABEL_TO_INDEX[row["binary_label"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary PlantVillage preprocessing, classical baseline training, and neural-network training."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Deduplicate the PlantVillage archive, create binary labels, resize images, and write train/val/test splits.",
    )
    preprocess_parser.add_argument("--archive", required=True, type=Path, help="Path to the PlantVillage zip archive.")
    preprocess_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Empty directory where the preprocessed split dataset will be written.",
    )
    preprocess_parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Fraction of the dataset reserved for testing. Default: {DEFAULT_TEST_SIZE}",
    )
    preprocess_parser.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help=f"Fraction of the dataset reserved for validation. Default: {DEFAULT_VAL_SIZE}",
    )
    preprocess_parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed used for the stratified split. Default: {DEFAULT_RANDOM_STATE}",
    )
    preprocess_parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Square output image size in pixels. Default: {DEFAULT_IMAGE_SIZE}",
    )

    train_parser = subparsers.add_parser(
        "train-baseline",
        help="Train and evaluate an HOG + Linear SVM baseline from a preprocessed manifest.",
    )
    train_parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the preprocessing manifest.csv file.",
    )
    train_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where baseline metrics, confusion matrix, and summary files will be written.",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for the LinearSVC model. Default: {DEFAULT_RANDOM_STATE}",
    )

    cnn_parser = subparsers.add_parser(
        "train-cnn",
        help="Train and evaluate a small custom CNN baseline.",
    )
    cnn_parser.add_argument("--manifest", required=True, type=Path, help="Path to the preprocessing manifest.csv file.")
    cnn_parser.add_argument("--output-dir", required=True, type=Path, help="Directory where CNN artifacts will be written.")
    cnn_parser.add_argument("--epochs", type=int, default=DEFAULT_CNN_EPOCHS, help=f"Training epochs. Default: {DEFAULT_CNN_EPOCHS}")
    cnn_parser.add_argument("--batch-size", type=int, default=DEFAULT_CNN_BATCH_SIZE, help=f"Batch size. Default: {DEFAULT_CNN_BATCH_SIZE}")
    cnn_parser.add_argument("--learning-rate", type=float, default=DEFAULT_CNN_LR, help=f"Adam learning rate. Default: {DEFAULT_CNN_LR}")
    cnn_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help=f"Random seed. Default: {DEFAULT_RANDOM_STATE}")
    cnn_parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count. Default: 0")
    cnn_parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or mps. Default: auto")
    cnn_parser.add_argument("--no-plot", action="store_true", help="Skip writing the training curve PNG.")

    resnet_parser = subparsers.add_parser(
        "train-resnet18",
        help="Train and evaluate a ResNet-18 comparison model.",
    )
    resnet_parser.add_argument("--manifest", required=True, type=Path, help="Path to the preprocessing manifest.csv file.")
    resnet_parser.add_argument("--output-dir", required=True, type=Path, help="Directory where ResNet-18 artifacts will be written.")
    resnet_parser.add_argument("--epochs", type=int, default=DEFAULT_RESNET_EPOCHS, help=f"Training epochs. Default: {DEFAULT_RESNET_EPOCHS}")
    resnet_parser.add_argument("--batch-size", type=int, default=DEFAULT_RESNET_BATCH_SIZE, help=f"Batch size. Default: {DEFAULT_RESNET_BATCH_SIZE}")
    resnet_parser.add_argument("--learning-rate", type=float, default=DEFAULT_RESNET_LR, help=f"AdamW learning rate. Default: {DEFAULT_RESNET_LR}")
    resnet_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help=f"Random seed. Default: {DEFAULT_RANDOM_STATE}")
    resnet_parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count. Default: 0")
    resnet_parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or mps. Default: auto")
    resnet_parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ResNet-18 feature layers and train only the classifier head.",
    )
    resnet_parser.add_argument(
        "--fine-tune-last-block",
        action="store_true",
        help="Unfreeze ResNet layer4 in addition to the classifier head.",
    )
    resnet_parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ImageNet pretrained weights for a transfer-learning comparison. Default: train from scratch.",
    )
    resnet_parser.add_argument("--no-plot", action="store_true", help="Skip writing the training curve PNG.")

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run single-image inference from a saved neural-network checkpoint.",
    )
    predict_parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a saved checkpoint.pt file.")
    predict_parser.add_argument("--image", required=True, type=Path, help="Path to an image to classify.")
    predict_parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda, or mps. Default: auto")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "preprocess":
            preprocess_dataset(
                archive_path=args.archive,
                output_dir=args.output_dir,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_state,
                image_size=args.image_size,
            )
        elif args.command == "train-baseline":
            train_baseline(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                random_state=args.random_state,
            )
        elif args.command == "train-cnn":
            train_neural_model(
                model_name="custom_cnn",
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                random_state=args.random_state,
                num_workers=args.num_workers,
                requested_device=args.device,
                write_plot=not args.no_plot,
            )
        elif args.command == "train-resnet18":
            train_neural_model(
                model_name="resnet18",
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                random_state=args.random_state,
                num_workers=args.num_workers,
                requested_device=args.device,
                write_plot=not args.no_plot,
                pretrained=args.pretrained,
                freeze_backbone=args.freeze_backbone,
                fine_tune_last_block=args.fine_tune_last_block,
            )
        elif args.command == "predict":
            predict_image(
                checkpoint_path=args.checkpoint,
                image_path=args.image,
                requested_device=args.device,
            )
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:  # pragma: no cover - surfaced via CLI
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def preprocess_dataset(
    archive_path: Path,
    output_dir: Path,
    test_size: float,
    val_size: float,
    random_state: int,
    image_size: int,
) -> None:
    validate_preprocess_arguments(
        archive_path=archive_path,
        output_dir=output_dir,
        test_size=test_size,
        val_size=val_size,
        image_size=image_size,
    )

    print("Indexing archive and removing duplicate paths...")
    records, duplicates_removed = collect_unique_records(archive_path)
    if not records:
        raise ValueError("No supported image files were found in the archive.")

    print(f"Found {len(records):,} unique images after removing {duplicates_removed:,} duplicate archive entries.")
    train_records, val_records, test_records = split_records(
        records,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    split_map = {"train": train_records}
    if val_records:
        split_map["val"] = val_records
    split_map["test"] = test_records

    print("Writing resized split dataset...")
    manifest_rows = write_preprocessed_images(
        archive_path=archive_path,
        output_dir=output_dir,
        split_map=split_map,
        image_size=image_size,
    )

    manifest_path = output_dir / MANIFEST_NAME
    write_manifest(manifest_path, manifest_rows)

    preprocess_summary = build_preprocess_summary(
        archive_path=archive_path,
        output_dir=output_dir,
        records=records,
        split_map=split_map,
        duplicates_removed=duplicates_removed,
        image_size=image_size,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    write_json(output_dir / PREPROCESS_SUMMARY_NAME, preprocess_summary)
    verify_preprocess_outputs(manifest_rows, image_size=image_size)

    print(f"Manifest written to {manifest_path}")
    print(f"Summary written to {output_dir / PREPROCESS_SUMMARY_NAME}")
    print(f"Train images: {len(train_records):,}")
    if val_records:
        print(f"Val images:   {len(val_records):,}")
    print(f"Test images:  {len(test_records):,}")


def train_baseline(manifest_path: Path, output_dir: Path, random_state: int) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    dataset_root = manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not test_rows:
        raise ValueError("Manifest must contain both train and test rows.")

    print("Extracting HOG features for the training split...")
    train_features, y_train = build_feature_matrix(train_rows, dataset_root)
    print("Extracting HOG features for the test split...")
    test_features, y_test = build_feature_matrix(test_rows, dataset_root)
    image_size = infer_preprocessed_image_size(train_rows, dataset_root)

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=10000,
                    dual="auto",
                ),
            ),
        ]
    )

    print("Training LinearSVC baseline...")
    classifier.fit(train_features, y_train)

    decision_scores = classifier.decision_function(test_features)
    y_pred = classifier.predict(test_features)

    metrics_payload = build_metrics_payload(
        rows=rows,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        y_test=y_test,
        y_pred=y_pred,
        decision_scores=decision_scores,
        feature_length=int(train_features.shape[1]),
        image_size=image_size,
        random_state=random_state,
    )

    metrics_path = output_dir / METRICS_NAME
    confusion_matrix_path = output_dir / CONFUSION_MATRIX_NAME
    summary_path = output_dir / SUMMARY_NAME

    write_json(metrics_path, metrics_payload)
    write_confusion_matrix_csv(confusion_matrix_path, metrics_payload["confusion_matrix"])
    summary_text = format_summary(metrics_payload)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"\nMetrics written to {metrics_path}")
    print(f"Confusion matrix written to {confusion_matrix_path}")
    print(f"Summary written to {summary_path}")


def validate_preprocess_arguments(
    archive_path: Path,
    output_dir: Path,
    test_size: float,
    val_size: float,
    image_size: int,
) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty so stale files cannot leak into a new split: {output_dir}"
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0.0 <= val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1, or 0 to disable validation output.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1 so training records remain.")
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")


def collect_unique_records(archive_path: Path) -> tuple[list[ImageRecord], int]:
    records: list[ImageRecord] = []
    seen_relpaths: set[str] = set()
    duplicates_removed = 0

    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            parsed = normalize_archive_member(member.filename)
            if parsed is None:
                continue
            source_class, normalized_relpath = parsed
            if normalized_relpath in seen_relpaths:
                duplicates_removed += 1
                continue
            seen_relpaths.add(normalized_relpath)
            records.append(
                ImageRecord(
                    archive_member=member.filename,
                    normalized_relpath=normalized_relpath,
                    source_class=source_class,
                    binary_label=map_binary_label(source_class),
                )
            )

    records.sort(key=lambda record: record.normalized_relpath)
    return records, duplicates_removed


def normalize_archive_member(member_name: str) -> tuple[str, str] | None:
    path = PurePosixPath(member_name)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None

    parts = path.parts
    if len(parts) >= 3 and parts[0].lower() == "plantvillage" and parts[1] == "PlantVillage":
        relative_parts = parts[2:]
    elif len(parts) >= 2 and parts[0] == "PlantVillage":
        relative_parts = parts[1:]
    else:
        return None

    if len(relative_parts) < 2:
        return None

    source_class = relative_parts[0]
    normalized_relpath = "/".join(relative_parts)
    return source_class, normalized_relpath


def map_binary_label(source_class: str) -> str:
    return NEGATIVE_LABEL if "healthy" in source_class.lower() else POSITIVE_LABEL


def split_records(
    records: list[ImageRecord],
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    stratify_labels = [record.source_class for record in records]
    train_val_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    if val_size > 0:
        relative_val_size = val_size / (1.0 - test_size)
        train_records, val_records = train_test_split(
            train_val_records,
            test_size=relative_val_size,
            random_state=random_state + 1,
            stratify=[record.source_class for record in train_val_records],
        )
    else:
        train_records = train_val_records
        val_records = []

    train_records = sorted(train_records, key=lambda record: record.normalized_relpath)
    val_records = sorted(val_records, key=lambda record: record.normalized_relpath)
    test_records = sorted(test_records, key=lambda record: record.normalized_relpath)
    return train_records, val_records, test_records


def write_preprocessed_images(
    archive_path: Path,
    output_dir: Path,
    split_map: dict[str, list[ImageRecord]],
    image_size: int,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    resample = Image.Resampling.BILINEAR

    with ZipFile(archive_path) as archive:
        total_images = sum(len(records) for records in split_map.values())
        processed = 0
        for split, records in split_map.items():
            for record in records:
                processed += 1
                if processed % 1000 == 0 or processed == total_images:
                    print(f"Processed {processed:,}/{total_images:,} images...")

                stored_relpath = Path(split) / record.binary_label / build_output_name(record)
                destination = output_dir / stored_relpath
                destination.parent.mkdir(parents=True, exist_ok=True)

                with archive.open(record.archive_member) as image_stream:
                    try:
                        with Image.open(image_stream) as image:
                            processed_image = image.convert("RGB").resize((image_size, image_size), resample=resample)
                            processed_image.save(destination, format="PNG")
                    except UnidentifiedImageError as exc:
                        raise ValueError(f"Unsupported image in archive: {record.archive_member}") from exc

                manifest_rows.append(
                    {
                        "split": split,
                        "binary_label": record.binary_label,
                        "source_class": record.source_class,
                        "archive_member": record.archive_member,
                        "normalized_relpath": record.normalized_relpath,
                        "stored_relpath": stored_relpath.as_posix(),
                    }
                )

    return manifest_rows


def build_output_name(record: ImageRecord) -> str:
    original_name = PurePosixPath(record.normalized_relpath).stem
    source_component = slugify(record.source_class)
    name_component = slugify(original_name)
    digest = hashlib.sha1(record.normalized_relpath.encode("utf-8")).hexdigest()[:10]
    return f"{source_component}__{name_component}__{digest}.png"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "image"


def write_manifest(manifest_path: Path, rows: Iterable[dict[str, str]]) -> None:
    fieldnames = [
        "split",
        "binary_label",
        "source_class",
        "archive_member",
        "normalized_relpath",
        "stored_relpath",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_preprocess_summary(
    archive_path: Path,
    output_dir: Path,
    records: list[ImageRecord],
    split_map: dict[str, list[ImageRecord]],
    duplicates_removed: int,
    image_size: int,
    test_size: float,
    val_size: float,
    random_state: int,
) -> dict[str, object]:
    unique_records = len(records)
    binary_counts = Counter(record.binary_label for record in records)
    source_counts = Counter(record.source_class for record in records)
    split_counts = {
        split: dict(Counter(record.binary_label for record in split_records))
        for split, split_records in split_map.items()
    }
    source_split_counts = {
        split: dict(Counter(record.source_class for record in split_records))
        for split, split_records in split_map.items()
    }
    return {
        "archive_path": str(archive_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "unique_records": unique_records,
        "duplicates_removed": duplicates_removed,
        "image_size": image_size,
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "binary_counts": dict(binary_counts),
        "source_class_counts": dict(source_counts),
        "split_counts": split_counts,
        "source_class_split_counts": source_split_counts,
        "imbalance_ratio": round(binary_counts[POSITIVE_LABEL] / binary_counts[NEGATIVE_LABEL], 4),
    }


def verify_preprocess_outputs(manifest_rows: list[dict[str, str]], image_size: int) -> None:
    if len(manifest_rows) != len({row["normalized_relpath"] for row in manifest_rows}):
        raise ValueError("Duplicate normalized_relpath values detected in the manifest.")

    split_paths = {
        split: {row["stored_relpath"] for row in manifest_rows if row["split"] == split}
        for split in {row["split"] for row in manifest_rows}
    }
    seen_paths: set[str] = set()
    for split, paths in split_paths.items():
        overlap = seen_paths & paths
        if overlap:
            raise ValueError(f"Split {split} overlaps another split for {len(overlap)} files.")
        seen_paths.update(paths)

    for row in manifest_rows:
        expected_label = map_binary_label(row["source_class"])
        if row["binary_label"] != expected_label:
            raise ValueError(
                "Manifest label mapping does not match the healthy vs diseased rule "
                f"for source class {row['source_class']}."
            )

    if image_size <= 0:
        raise ValueError("image_size must remain positive.")


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty.")
    required_columns = {
        "split",
        "binary_label",
        "source_class",
        "archive_member",
        "normalized_relpath",
        "stored_relpath",
    }
    if not required_columns.issubset(reader.fieldnames or set()):
        raise ValueError(f"Manifest is missing required columns: {sorted(required_columns)}")
    return rows


def build_feature_matrix(rows: list[dict[str, str]], dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage.feature import hog
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "scikit-image is required for HOG features. Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    features: list[np.ndarray] = []
    labels: list[int] = []

    for index, row in enumerate(rows, start=1):
        image_path = dataset_root / row["stored_relpath"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing preprocessed image referenced by manifest: {image_path}")

        with Image.open(image_path) as image:
            grayscale = image.convert("L")
            image_array = np.asarray(grayscale, dtype=np.float32) / 255.0

        descriptor = hog(
            image_array,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        ).astype(np.float32, copy=False)
        features.append(descriptor)
        labels.append(1 if row["binary_label"] == POSITIVE_LABEL else 0)

        if index % 2000 == 0 or index == len(rows):
            print(f"  Extracted features for {index:,}/{len(rows):,} images...")

    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def infer_preprocessed_image_size(rows: list[dict[str, str]], dataset_root: Path) -> list[int]:
    sample_path = dataset_root / rows[0]["stored_relpath"]
    with Image.open(sample_path) as image:
        return [image.width, image.height]


def build_metrics_payload(
    rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray,
    feature_length: int,
    image_size: list[int],
    random_state: int,
) -> dict[str, object]:
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, decision_scores)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    return {
        "dataset": {
            "total_records": len(rows),
            "train_records": len(train_rows),
            "val_records": len(val_rows),
            "test_records": len(test_rows),
            "binary_counts": dict(Counter(row["binary_label"] for row in rows)),
            "train_binary_counts": dict(Counter(row["binary_label"] for row in train_rows)),
            "val_binary_counts": dict(Counter(row["binary_label"] for row in val_rows)),
            "test_binary_counts": dict(Counter(row["binary_label"] for row in test_rows)),
        },
        "preprocessing": {
            "image_mode": "RGB",
            "grayscale_for_features": True,
            "image_size": image_size,
            "binary_label_rule": "Folders containing 'healthy' map to healthy; all others map to diseased.",
        },
        "feature_extractor": {
            "name": "HOG",
            "orientations": 9,
            "pixels_per_cell": [8, 8],
            "cells_per_block": [2, 2],
            "block_norm": "L2-Hys",
            "transform_sqrt": True,
            "feature_length": feature_length,
        },
        "classifier": {
            "name": "LinearSVC",
            "class_weight": "balanced",
            "random_state": random_state,
            "positive_label": POSITIVE_LABEL,
        },
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision_diseased": precision,
            "recall_diseased": recall,
            "f1_diseased": f1,
            "roc_auc": roc_auc,
        },
        "confusion_matrix": {
            "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
            "matrix": matrix.tolist(),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def format_summary(metrics_payload: dict[str, object]) -> str:
    dataset = metrics_payload["dataset"]
    feature_extractor = metrics_payload["feature_extractor"]
    metrics = metrics_payload["metrics"]
    confusion = metrics_payload["confusion_matrix"]
    image_width, image_height = metrics_payload["preprocessing"]["image_size"]

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines = [
        "# Binary Plant Disease Baseline",
        "",
        "## Methodology",
        f"- Dataset: {dataset['total_records']:,} unique PlantVillage images after deduplicating the archive's mirrored directory trees.",
        "- Labels: folders containing `healthy` are mapped to `healthy`; every other original PlantVillage folder is mapped to `diseased`.",
        (
            f"- Split: manifest contains {dataset['train_records']:,} train, {dataset.get('val_records', 0):,} validation, "
            f"and {dataset['test_records']:,} test images; this classical baseline trains on train and evaluates on test."
        ),
        f"- Preprocessing: images are resized to {image_width}x{image_height} RGB PNG files, then converted to grayscale during feature extraction.",
        (
            "- Features and model: Histogram of Oriented Gradients "
            f"(orientations={feature_extractor['orientations']}, pixels_per_cell={tuple(feature_extractor['pixels_per_cell'])}, "
            f"cells_per_block={tuple(feature_extractor['cells_per_block'])}) followed by `StandardScaler` and "
            "`LinearSVC(class_weight=\"balanced\")`."
        ),
        "",
        "## Evaluation Metrics",
        "- Accuracy measures overall correctness.",
        "- Balanced accuracy adjusts for the strong healthy vs diseased class imbalance.",
        "- Precision, recall, and F1 are reported for the `diseased` class so missed diseased leaves are visible.",
        "- ROC-AUC is computed from the SVM decision scores.",
        "- The confusion matrix is included to show false positives and false negatives explicitly.",
        "",
        "## Baseline Performance",
        f"- Accuracy: {pct(metrics['accuracy'])}",
        f"- Balanced accuracy: {pct(metrics['balanced_accuracy'])}",
        f"- Precision (`diseased`): {pct(metrics['precision_diseased'])}",
        f"- Recall (`diseased`): {pct(metrics['recall_diseased'])}",
        f"- F1 (`diseased`): {pct(metrics['f1_diseased'])}",
        f"- ROC-AUC: {pct(metrics['roc_auc'])}",
        (
            "- Confusion matrix (`actual x predicted`, labels ordered as "
            f"`[{NEGATIVE_LABEL}, {POSITIVE_LABEL}]`): {confusion['matrix']}"
        ),
        "",
        (
            "This baseline is intentionally simple and reproducible, but PlantVillage is a controlled laboratory-style "
            "dataset. Strong held-out performance here can still overestimate how well the same model will generalize "
            "to real field images with different lighting, backgrounds, and camera quality."
        ),
    ]
    return "\n".join(lines)


def train_neural_model(
    model_name: str,
    manifest_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
    num_workers: int,
    requested_device: str,
    write_plot: bool,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    fine_tune_last_block: bool = False,
) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    validate_neural_training_arguments(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, num_workers=num_workers)

    torch, nn, data_loader_cls, models, transforms = import_torch_stack()
    set_reproducibility(torch, random_state)
    device = resolve_device(torch, requested_device)

    rows = read_manifest(manifest_path)
    train_rows, val_rows, test_rows = get_neural_split_rows(rows)
    dataset_root = manifest_path.parent
    image_size = infer_preprocessed_image_size(train_rows, dataset_root)

    train_transform = build_image_transform(transforms, model_name=model_name, train=True, image_size=image_size[0])
    eval_transform = build_image_transform(transforms, model_name=model_name, train=False, image_size=image_size[0])
    generator = torch.Generator()
    generator.manual_seed(random_state)

    train_loader = data_loader_cls(
        ManifestImageDataset(train_rows, dataset_root, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = data_loader_cls(
        ManifestImageDataset(val_rows, dataset_root, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = data_loader_cls(
        ManifestImageDataset(test_rows, dataset_root, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model, architecture = build_neural_architecture(
        model_name=model_name,
        nn=nn,
        models=models,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        fine_tune_last_block=fine_tune_last_block,
    )
    model.to(device)

    class_weights = build_class_weights(torch, train_rows, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

    if model_name == "resnet18":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
        optimizer_name = "AdamW"
    else:
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
        optimizer_name = "Adam"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training {architecture['display_name']} on {device}...")
    print(f"Train/val/test records: {len(train_rows):,}/{len(val_rows):,}/{len(test_rows):,}")

    history: list[dict[str, object]] = []
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = run_neural_epoch(
            torch=torch,
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_neural_epoch(
            torch=torch,
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        history_row = build_history_row(epoch, train_metrics, val_metrics)
        history.append(history_row)

        val_loss = float(val_metrics["loss"])
        val_f1 = float(val_metrics["f1_diseased"])
        is_better = val_loss < best_val_loss - 1e-8 or (abs(val_loss - best_val_loss) <= 1e-8 and val_f1 > best_val_f1)
        if is_better:
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1_diseased']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = run_neural_epoch(
        torch=torch,
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    checkpoint_payload = {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "label_to_index": LABEL_TO_INDEX,
        "index_to_label": INDEX_TO_LABEL,
        "image_size": image_size,
        "architecture": architecture,
        "normalization": get_normalization(model_name),
        "pretrained": pretrained if model_name == "resnet18" else False,
        "freeze_backbone": freeze_backbone if model_name == "resnet18" else False,
        "fine_tune_last_block": fine_tune_last_block if model_name == "resnet18" else False,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint_payload, output_dir / CHECKPOINT_NAME)

    metrics_payload = build_neural_metrics_payload(
        rows=rows,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        model_name=model_name,
        architecture=architecture,
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        random_state=random_state,
        device=str(device),
        class_weights=[float(value) for value in class_weights.detach().cpu().tolist()],
        best_epoch=best_epoch,
        test_metrics=test_metrics,
        history=history,
    )

    write_json(output_dir / METRICS_NAME, metrics_payload)
    write_confusion_matrix_csv(output_dir / CONFUSION_MATRIX_NAME, metrics_payload["confusion_matrix"])
    write_training_history_csv(output_dir / HISTORY_NAME, history)
    if write_plot:
        write_training_plot(output_dir / PLOT_NAME, history)
    summary_text = format_neural_summary(metrics_payload)
    (output_dir / SUMMARY_NAME).write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"\nCheckpoint written to {output_dir / CHECKPOINT_NAME}")
    print(f"Metrics written to {output_dir / METRICS_NAME}")
    print(f"Confusion matrix written to {output_dir / CONFUSION_MATRIX_NAME}")
    print(f"Training history written to {output_dir / HISTORY_NAME}")
    if write_plot:
        print(f"Training plot written to {output_dir / PLOT_NAME}")
    print(f"Summary written to {output_dir / SUMMARY_NAME}")


def validate_neural_training_arguments(epochs: int, batch_size: int, learning_rate: float, num_workers: int) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")


def import_torch_stack():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "PyTorch, torchvision, and their dependencies are required for neural-network commands. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc
    return torch, nn, DataLoader, models, transforms


def set_reproducibility(torch, random_state: int) -> None:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def resolve_device(torch, requested_device: str):
    requested = requested_device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS was requested, but this PyTorch install does not report MPS availability.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")
    return torch.device(requested)


def get_neural_split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not val_rows or not test_rows:
        raise ValueError(
            "Neural-network training requires train, val, and test rows. "
            "Regenerate preprocessing with `--val-size 0.1 --image-size 224`."
        )
    for split_name, split_rows in {"train": train_rows, "val": val_rows, "test": test_rows}.items():
        labels = {row["binary_label"] for row in split_rows}
        if labels != {NEGATIVE_LABEL, POSITIVE_LABEL}:
            raise ValueError(f"Split {split_name} must contain both healthy and diseased records.")
    return train_rows, val_rows, test_rows


def build_image_transform(transforms, model_name: str, train: bool, image_size: int | None = None):
    normalization = get_normalization(model_name)
    transform_steps = []
    if image_size is not None:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    if train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization["mean"], std=normalization["std"]),
        ]
    )
    return transforms.Compose(transform_steps)


def get_normalization(model_name: str) -> dict[str, list[float]]:
    if model_name == "resnet18":
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    return {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }


def build_neural_architecture(
    model_name: str,
    nn,
    models,
    pretrained: bool,
    freeze_backbone: bool,
    fine_tune_last_block: bool,
):
    if model_name == "custom_cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        architecture = {
            "name": "custom_cnn",
            "display_name": "Custom CNN",
            "input_shape": [3, 224, 224],
            "blocks": [
                "Conv2d(3, 32, 3, padding=1) -> ReLU -> BatchNorm2d -> MaxPool2d",
                "Conv2d(32, 64, 3, padding=1) -> ReLU -> BatchNorm2d -> MaxPool2d",
                "Conv2d(64, 128, 3, padding=1) -> ReLU -> BatchNorm2d -> MaxPool2d",
                "AdaptiveAvgPool2d -> Flatten -> Dropout(0.3) -> Linear(128, 2)",
            ],
        }
        return model, architecture

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features

        if freeze_backbone or fine_tune_last_block:
            for parameter in model.parameters():
                parameter.requires_grad = False
        if fine_tune_last_block:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
        model.fc = nn.Linear(in_features, 2)

        architecture = {
            "name": "resnet18",
            "display_name": "ResNet-18 Transfer Learning",
            "input_shape": [3, 224, 224],
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "fine_tune_last_block": fine_tune_last_block,
            "classifier_head": f"Linear({in_features}, 2)",
        }
        return model, architecture

    raise ValueError(f"Unsupported neural model: {model_name}")


def build_class_weights(torch, train_rows: list[dict[str, str]], device):
    counts = Counter(row["binary_label"] for row in train_rows)
    missing = [label for label in LABEL_TO_INDEX if counts[label] == 0]
    if missing:
        raise ValueError(f"Cannot compute class weights because training split is missing: {missing}")
    total = sum(counts.values())
    weights = [
        total / (len(LABEL_TO_INDEX) * counts[INDEX_TO_LABEL[index]])
        for index in range(len(LABEL_TO_INDEX))
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_neural_epoch(torch, model, data_loader, criterion, device, optimizer):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    positive_scores: list[float] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = int(labels.size(0))
            total_examples += batch_size
            total_loss += float(loss.item()) * batch_size
            probabilities = torch.softmax(logits.detach(), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
            positive_scores.extend(probabilities[:, LABEL_TO_INDEX[POSITIVE_LABEL]].detach().cpu().tolist())

    if total_examples == 0:
        raise ValueError("DataLoader produced no examples.")
    return build_classification_metrics(
        y_true=np.asarray(y_true, dtype=np.int32),
        y_pred=np.asarray(y_pred, dtype=np.int32),
        positive_scores=np.asarray(positive_scores, dtype=np.float32),
        loss=total_loss / total_examples,
    )


def build_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_scores: np.ndarray | None = None,
    loss: float | None = None,
) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=[LABEL_TO_INDEX[NEGATIVE_LABEL], LABEL_TO_INDEX[POSITIVE_LABEL]])
    tn, fp, fn, tp = matrix.ravel()
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_diseased": float(precision_score(y_true, y_pred, pos_label=LABEL_TO_INDEX[POSITIVE_LABEL], zero_division=0)),
        "recall_diseased": float(recall_score(y_true, y_pred, pos_label=LABEL_TO_INDEX[POSITIVE_LABEL], zero_division=0)),
        "f1_diseased": float(f1_score(y_true, y_pred, pos_label=LABEL_TO_INDEX[POSITIVE_LABEL], zero_division=0)),
        "confusion_matrix": {
            "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
            "matrix": matrix.tolist(),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    if positive_scores is not None and len(set(y_true.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
    else:
        metrics["roc_auc"] = None
    return metrics


def build_history_row(epoch: int, train_metrics: dict[str, object], val_metrics: dict[str, object]) -> dict[str, object]:
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_f1_diseased": train_metrics["f1_diseased"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision_diseased": val_metrics["precision_diseased"],
        "val_recall_diseased": val_metrics["recall_diseased"],
        "val_f1_diseased": val_metrics["f1_diseased"],
    }


def build_neural_metrics_payload(
    rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    model_name: str,
    architecture: dict[str, object],
    image_size: list[int],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    optimizer_name: str,
    random_state: int,
    device: str,
    class_weights: list[float],
    best_epoch: int,
    test_metrics: dict[str, object],
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "dataset": {
            "total_records": len(rows),
            "train_records": len(train_rows),
            "val_records": len(val_rows),
            "test_records": len(test_rows),
            "binary_counts": dict(Counter(row["binary_label"] for row in rows)),
            "train_binary_counts": dict(Counter(row["binary_label"] for row in train_rows)),
            "val_binary_counts": dict(Counter(row["binary_label"] for row in val_rows)),
            "test_binary_counts": dict(Counter(row["binary_label"] for row in test_rows)),
        },
        "preprocessing": {
            "image_mode": "RGB",
            "image_size": image_size,
            "normalization": get_normalization(model_name),
            "binary_label_rule": "Folders containing 'healthy' map to healthy; all others map to diseased.",
        },
        "model": architecture,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "loss": "CrossEntropyLoss",
            "class_weights": class_weights,
            "best_epoch": best_epoch,
            "random_state": random_state,
            "device": device,
        },
        "metrics": {
            key: value
            for key, value in test_metrics.items()
            if key != "confusion_matrix"
        },
        "confusion_matrix": test_metrics["confusion_matrix"],
        "history": history,
    }


def write_training_history_csv(path: Path, history: list[dict[str, object]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_f1_diseased",
        "val_loss",
        "val_accuracy",
        "val_precision_diseased",
        "val_recall_diseased",
        "val_f1_diseased",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_training_plot(path: Path, history: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"Warning: could not write training plot because matplotlib is unavailable: {exc}", file=sys.stderr)
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_accuracy"]) for row in history]
    val_acc = [float(row["val_accuracy"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_neural_summary(metrics_payload: dict[str, object]) -> str:
    dataset = metrics_payload["dataset"]
    model = metrics_payload["model"]
    training = metrics_payload["training"]
    metrics = metrics_payload["metrics"]
    confusion = metrics_payload["confusion_matrix"]
    image_width, image_height = metrics_payload["preprocessing"]["image_size"]

    def pct(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.2f}%"

    lines = [
        f"# Binary Plant Disease {model['display_name']}",
        "",
        "## Methodology",
        f"- Dataset: {dataset['total_records']:,} PlantVillage images split into {dataset['train_records']:,} train, {dataset['val_records']:,} validation, and {dataset['test_records']:,} test images.",
        "- Labels: folders containing `healthy` are mapped to `healthy`; every other original PlantVillage folder is mapped to `diseased`.",
        f"- Preprocessing: images are resized to {image_width}x{image_height} RGB PNG files and normalized before training.",
        f"- Model: {model['display_name']} with a two-class output head.",
        f"- Training: {training['optimizer']} optimizer, weighted cross-entropy loss, batch size {training['batch_size']}, best checkpoint selected at epoch {training['best_epoch']}.",
        "",
        "## Test Performance",
        f"- Accuracy: {pct(metrics['accuracy'])}",
        f"- Balanced accuracy: {pct(metrics['balanced_accuracy'])}",
        f"- Precision (`diseased`): {pct(metrics['precision_diseased'])}",
        f"- Recall (`diseased`): {pct(metrics['recall_diseased'])}",
        f"- F1 (`diseased`): {pct(metrics['f1_diseased'])}",
        f"- ROC-AUC: {pct(metrics['roc_auc'])}",
        (
            "- Confusion matrix (`actual x predicted`, labels ordered as "
            f"`[{NEGATIVE_LABEL}, {POSITIVE_LABEL}]`): {confusion['matrix']}"
        ),
        "",
        (
            "This neural-network result should be compared with the HOG + LinearSVC baseline, while remembering that "
            "PlantVillage images are controlled lab-style images and may overestimate real field performance."
        ),
    ]
    return "\n".join(lines)


def predict_image(checkpoint_path: Path, image_path: Path, requested_device: str) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    torch, nn, _data_loader_cls, models, transforms = import_torch_stack()
    device = resolve_device(torch, requested_device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    architecture = checkpoint.get("architecture", {})

    model, _ = build_neural_architecture(
        model_name=model_name,
        nn=nn,
        models=models,
        pretrained=False,
        freeze_backbone=bool(architecture.get("freeze_backbone", False)),
        fine_tune_last_block=bool(architecture.get("fine_tune_last_block", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    checkpoint_image_size = checkpoint.get("image_size", [224, 224])[0]
    transform = build_image_transform(transforms, model_name=model_name, train=False, image_size=checkpoint_image_size)
    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    predicted_index = int(np.argmax(probabilities))
    predicted_label = INDEX_TO_LABEL[predicted_index]
    confidence = probabilities[predicted_index]
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"healthy: {probabilities[LABEL_TO_INDEX[NEGATIVE_LABEL]]:.4f}")
    print(f"diseased: {probabilities[LABEL_TO_INDEX[POSITIVE_LABEL]]:.4f}")


def write_confusion_matrix_csv(path: Path, payload: dict[str, object]) -> None:
    labels = payload["labels"]
    matrix = payload["matrix"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
