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
DEFAULT_IMAGE_SIZE = 64
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
POSITIVE_LABEL = "diseased"
NEGATIVE_LABEL = "healthy"
MANIFEST_NAME = "manifest.csv"
PREPROCESS_SUMMARY_NAME = "preprocess_summary.json"
METRICS_NAME = "metrics.json"
CONFUSION_MATRIX_NAME = "confusion_matrix.csv"
SUMMARY_NAME = "summary.md"


@dataclass(frozen=True)
class ImageRecord:
    archive_member: str
    normalized_relpath: str
    source_class: str
    binary_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary PlantVillage preprocessing and classical baseline training."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Deduplicate the PlantVillage archive, create binary labels, resize images, and write train/test splits.",
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

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "preprocess":
            preprocess_dataset(
                archive_path=args.archive,
                output_dir=args.output_dir,
                test_size=args.test_size,
                random_state=args.random_state,
                image_size=args.image_size,
            )
        elif args.command == "train-baseline":
            train_baseline(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                random_state=args.random_state,
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
    random_state: int,
    image_size: int,
) -> None:
    validate_preprocess_arguments(archive_path=archive_path, output_dir=output_dir, test_size=test_size, image_size=image_size)

    print("Indexing archive and removing duplicate paths...")
    records, duplicates_removed = collect_unique_records(archive_path)
    if not records:
        raise ValueError("No supported image files were found in the archive.")

    print(f"Found {len(records):,} unique images after removing {duplicates_removed:,} duplicate archive entries.")
    train_records, test_records = split_records(records, test_size=test_size, random_state=random_state)
    split_map = {"train": train_records, "test": test_records}

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
        random_state=random_state,
    )
    write_json(output_dir / PREPROCESS_SUMMARY_NAME, preprocess_summary)
    verify_preprocess_outputs(manifest_rows, image_size=image_size)

    print(f"Manifest written to {manifest_path}")
    print(f"Summary written to {output_dir / PREPROCESS_SUMMARY_NAME}")
    print(f"Train images: {len(train_records):,}")
    print(f"Test images:  {len(test_records):,}")


def train_baseline(manifest_path: Path, output_dir: Path, random_state: int) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    dataset_root = manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    train_rows = [row for row in rows if row["split"] == "train"]
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


def validate_preprocess_arguments(archive_path: Path, output_dir: Path, test_size: float, image_size: int) -> None:
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
    random_state: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    stratify_labels = [record.source_class for record in records]
    train_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    train_records = sorted(train_records, key=lambda record: record.normalized_relpath)
    test_records = sorted(test_records, key=lambda record: record.normalized_relpath)
    return train_records, test_records


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

    train_paths = {row["stored_relpath"] for row in manifest_rows if row["split"] == "train"}
    test_paths = {row["stored_relpath"] for row in manifest_rows if row["split"] == "test"}
    overlap = train_paths & test_paths
    if overlap:
        raise ValueError(f"Train and test splits overlap for {len(overlap)} files.")

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
            "test_records": len(test_rows),
            "binary_counts": dict(Counter(row["binary_label"] for row in rows)),
            "train_binary_counts": dict(Counter(row["binary_label"] for row in train_rows)),
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
        "- Split: fixed 80/20 train-test partition with `random_state=42`, stratified by original source folder so each plant-condition subtype remains represented.",
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
