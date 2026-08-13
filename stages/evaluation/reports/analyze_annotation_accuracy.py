"""
Analyze manually corrected cell annotation JSON files.

Outputs:
- summary.json
- per_class_metrics.csv
- image_error_rates.csv
- top_error_pairs.csv
- cell_level_records.csv
- confusion_matrix_counts.png
- confusion_matrix_normalized.png
- per_class_metrics.png
- top_error_pairs.png
- image_error_rate_top20.png
- confidence_correct_vs_incorrect.png
- analysis_report.md

Usage:
  python testing/analyze_cell_annotations.py \
    --artifacts-root results/heatmaps/image
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from stages.landcover_model.taxonomy.landcover_class_taxonomy import CANONICAL_CLASS_NAMES, canonicalize_label


def discover_annotation_files(artifacts_root: Path) -> List[Path]:
    return sorted(artifacts_root.rglob("*_annotations.json"), key=lambda p: str(p).lower())


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_records(annotation_files: List[Path]) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    issues = {
        "files_read": 0,
        "rows_total": 0,
        "rows_with_invalid_prediction": 0,
        "rows_with_invalid_correct_label": 0,
        "rows_incorrect_but_same_label": 0,
    }

    valid_classes = set(CANONICAL_CLASS_NAMES)

    for annotation_file in annotation_files:
        image_stem = annotation_file.name.replace("_annotations.json", "")
        with annotation_file.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)

        issues["files_read"] += 1
        for row in rows:
            issues["rows_total"] += 1

            raw_pred = row.get("model_prediction")
            raw_correct = row.get("correct_label")
            pred = canonicalize_label(raw_pred) or raw_pred
            correct = canonicalize_label(raw_correct) or raw_correct
            is_correct = bool(row.get("is_correct", True))

            if pred not in valid_classes:
                issues["rows_with_invalid_prediction"] += 1
                continue

            if correct not in valid_classes:
                issues["rows_with_invalid_correct_label"] += 1
                correct = pred

            # Ground truth for analytics comes from corrected label.
            true_label = correct

            if not is_correct and true_label == pred:
                # This signals an annotation state issue (checkbox changed but label unchanged).
                issues["rows_incorrect_but_same_label"] += 1

            records.append(
                {
                    "image": image_stem,
                    "cell_id": row.get("cell_id", ""),
                    "model_prediction": pred,
                    "correct_label": true_label,
                    "is_correct": is_correct,
                    "model_confidence": safe_float(row.get("model_confidence"), 0.0),
                    "raw_prediction": row.get("raw_prediction", pred),
                    "raw_confidence": safe_float(row.get("raw_confidence"), 0.0),
                    "switched_by_refinement": bool(row.get("switched_by_refinement", False)),
                }
            )

    return {"records": records, "issues": issues}


def compute_confusion(records: List[Dict[str, object]]) -> np.ndarray:
    n = len(CANONICAL_CLASS_NAMES)
    index = {name: idx for idx, name in enumerate(CANONICAL_CLASS_NAMES)}
    matrix = np.zeros((n, n), dtype=np.int64)

    for rec in records:
        true_idx = index[rec["correct_label"]]
        pred_idx = index[rec["model_prediction"]]
        matrix[true_idx, pred_idx] += 1

    return matrix


def compute_per_class_metrics(confusion: np.ndarray) -> List[Dict[str, object]]:
    metrics: List[Dict[str, object]] = []

    for i, class_name in enumerate(CANONICAL_CLASS_NAMES):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        support = int(confusion[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics.append(
            {
                "class": class_name,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return metrics


def compute_image_error_rates(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(lambda: {"total": 0, "incorrect": 0})

    for rec in records:
        grouped[rec["image"]]["total"] += 1
        if not rec["is_correct"]:
            grouped[rec["image"]]["incorrect"] += 1

    rows = []
    for image, counts in grouped.items():
        total = counts["total"]
        incorrect = counts["incorrect"]
        rows.append(
            {
                "image": image,
                "total_cells": total,
                "incorrect_cells": incorrect,
                "correct_cells": total - incorrect,
                "error_rate": (incorrect / total) if total > 0 else 0.0,
                "accuracy": ((total - incorrect) / total) if total > 0 else 0.0,
            }
        )

    rows.sort(key=lambda x: (-x["error_rate"], -x["incorrect_cells"], x["image"]))
    return rows


def compute_top_error_pairs(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    counter: Counter = Counter()

    for rec in records:
        pred = rec["model_prediction"]
        true = rec["correct_label"]
        if pred != true:
            counter[(pred, true)] += 1

    rows = []
    for (pred, true), count in counter.most_common():
        rows.append(
            {
                "predicted_as": pred,
                "actually": true,
                "count": int(count),
                "pair": f"{pred} -> {true}",
            }
        )
    return rows


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_confusion(confusion: np.ndarray, output_path: Path, normalized: bool = False) -> None:
    data = confusion.astype(np.float64)
    title_suffix = "Counts"

    if normalized:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)
        title_suffix = "Row-Normalized"

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CANONICAL_CLASS_NAMES)))
    ax.set_yticks(range(len(CANONICAL_CLASS_NAMES)))
    ax.set_xticklabels(CANONICAL_CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CANONICAL_CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Correct label")
    ax.set_title(f"Cell Annotation Confusion Matrix ({title_suffix})")

    # Keep text annotations readable.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = f"{val:.2f}" if normalized else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_metrics(metrics: List[Dict[str, object]], output_path: Path) -> None:
    classes = [m["class"] for m in metrics]
    precision = [m["precision"] for m in metrics]
    recall = [m["recall"] for m in metrics]
    f1 = [m["f1"] for m in metrics]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics from Manual Cell Annotations")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_error_pairs(error_pairs: List[Dict[str, object]], output_path: Path, top_k: int = 15) -> None:
    top = error_pairs[:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    if top:
        labels = [row["pair"] for row in top]
        counts = [row["count"] for row in top]
        y = np.arange(len(labels))
        ax.barh(y, counts)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No error pairs (all predictions matched corrections)", ha="center", va="center")

    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_k} Misclassification Pairs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_image_error_rates(image_rates: List[Dict[str, object]], output_path: Path, top_k: int = 20) -> None:
    top = image_rates[:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    if top:
        labels = [row["image"] for row in top]
        values = [row["error_rate"] for row in top]
        y = np.arange(len(labels))
        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No images found", ha="center", va="center")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Error Rate")
    ax.set_title(f"Top {top_k} Images by Cell Error Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confidence_distribution(records: List[Dict[str, object]], output_path: Path) -> None:
    correct_conf = [r["model_confidence"] for r in records if r["is_correct"]]
    incorrect_conf = [r["model_confidence"] for r in records if not r["is_correct"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    if correct_conf:
        ax.hist(correct_conf, bins=20, alpha=0.6, label="Correct cells")
    if incorrect_conf:
        ax.hist(incorrect_conf, bins=20, alpha=0.6, label="Incorrect cells")

    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("Cell Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Cells")
    ax.legend(loc="upper center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    output_path: Path,
    summary: Dict[str, object],
    per_class_metrics: List[Dict[str, object]],
    top_error_pairs: List[Dict[str, object]],
    image_rates: List[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# Cell Annotation Analysis Report")
    lines.append("")
    lines.append(f"Generated at: {summary['generated_at']}")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- Images analyzed: {summary['images_analyzed']}")
    lines.append(f"- Annotation files: {summary['annotation_files']}")
    lines.append(f"- Total cells: {summary['total_cells']}")
    lines.append(f"- Correct cells: {summary['correct_cells']}")
    lines.append(f"- Incorrect cells: {summary['incorrect_cells']}")
    lines.append(f"- Accuracy (from is_correct): {summary['accuracy_from_flag']:.4f}")
    lines.append(f"- Error rate (from is_correct): {summary['error_rate_from_flag']:.4f}")
    lines.append("")
    lines.append("## Data Quality Notes")
    lines.append(f"- Rows with invalid model prediction: {summary['issues']['rows_with_invalid_prediction']}")
    lines.append(f"- Rows with invalid correct label: {summary['issues']['rows_with_invalid_correct_label']}")
    lines.append(f"- Rows marked incorrect but unchanged label: {summary['issues']['rows_incorrect_but_same_label']}")
    lines.append("")
    lines.append("## Per-Class Metrics")
    lines.append("| Class | Support | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in per_class_metrics:
        lines.append(
            f"| {row['class']} | {row['support']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines.append("")
    lines.append("## Top Misclassification Pairs")
    if top_error_pairs:
        lines.append("| Predicted | Actually | Count |")
        lines.append("|---|---|---:|")
        for row in top_error_pairs[:15]:
            lines.append(f"| {row['predicted_as']} | {row['actually']} | {row['count']} |")
    else:
        lines.append("No misclassification pairs found.")
    lines.append("")
    lines.append("## Highest Error Images")
    if image_rates:
        lines.append("| Image | Cells | Incorrect | Error Rate |")
        lines.append("|---|---:|---:|---:|")
        for row in image_rates[:20]:
            lines.append(
                f"| {row['image']} | {row['total_cells']} | {row['incorrect_cells']} | {row['error_rate']:.3f} |"
            )
    else:
        lines.append("No images found.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze manual cell annotations and generate metrics + plots.")
    parser.add_argument(
        "--artifacts-root",
        default="results/heatmaps/image",
        help="Root directory containing per-image annotation artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Parent output directory. A timestamped child folder will be created.",
    )
    parser.add_argument(
        "--prefix",
        default="annotation_analysis",
        help="Prefix for timestamped output folder.",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.exists():
        raise FileNotFoundError(f"Artifacts root not found: {artifacts_root}")

    annotation_files = discover_annotation_files(artifacts_root)
    if not annotation_files:
        raise FileNotFoundError(f"No *_annotations.json files found under {artifacts_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"{args.prefix}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    parsed = parse_records(annotation_files)
    records: List[Dict[str, object]] = parsed["records"]
    issues: Dict[str, int] = parsed["issues"]

    if not records:
        raise RuntimeError("No valid records found after parsing annotations.")

    confusion = compute_confusion(records)
    per_class = compute_per_class_metrics(confusion)
    image_rates = compute_image_error_rates(records)
    top_pairs = compute_top_error_pairs(records)

    total_cells = len(records)
    correct_cells = sum(1 for r in records if r["is_correct"])
    incorrect_cells = total_cells - correct_cells

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts_root": str(artifacts_root.resolve()),
        "output_root": str(output_root.resolve()),
        "annotation_files": len(annotation_files),
        "images_analyzed": len({r["image"] for r in records}),
        "total_cells": total_cells,
        "correct_cells": correct_cells,
        "incorrect_cells": incorrect_cells,
        "accuracy_from_flag": (correct_cells / total_cells) if total_cells > 0 else 0.0,
        "error_rate_from_flag": (incorrect_cells / total_cells) if total_cells > 0 else 0.0,
        "issues": issues,
    }

    # Save machine-readable outputs.
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_csv(
        output_root / "per_class_metrics.csv",
        per_class,
        ["class", "support", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    save_csv(
        output_root / "image_error_rates.csv",
        image_rates,
        ["image", "total_cells", "incorrect_cells", "correct_cells", "error_rate", "accuracy"],
    )
    save_csv(
        output_root / "top_error_pairs.csv",
        top_pairs,
        ["predicted_as", "actually", "count", "pair"],
    )
    save_csv(
        output_root / "cell_level_records.csv",
        records,
        [
            "image",
            "cell_id",
            "model_prediction",
            "correct_label",
            "is_correct",
            "model_confidence",
            "raw_prediction",
            "raw_confidence",
            "switched_by_refinement",
        ],
    )

    # Save visual outputs.
    plot_confusion(confusion, output_root / "confusion_matrix_counts.png", normalized=False)
    plot_confusion(confusion, output_root / "confusion_matrix_normalized.png", normalized=True)
    plot_per_class_metrics(per_class, output_root / "per_class_metrics.png")
    plot_top_error_pairs(top_pairs, output_root / "top_error_pairs.png", top_k=15)
    plot_image_error_rates(image_rates, output_root / "image_error_rate_top20.png", top_k=20)
    plot_confidence_distribution(records, output_root / "confidence_correct_vs_incorrect.png")

    # Save a human-readable markdown report.
    write_markdown_report(output_root / "analysis_report.md", summary, per_class, top_pairs, image_rates)

    print("=" * 72)
    print("Cell Annotation Analysis Complete")
    print("=" * 72)
    print(f"Annotation files: {len(annotation_files)}")
    print(f"Images analyzed: {summary['images_analyzed']}")
    print(f"Total cells: {summary['total_cells']}")
    print(f"Accuracy: {summary['accuracy_from_flag']:.4f}")
    print(f"Error rate: {summary['error_rate_from_flag']:.4f}")
    print(f"Output folder: {output_root.resolve()}")


if __name__ == "__main__":
    main()

