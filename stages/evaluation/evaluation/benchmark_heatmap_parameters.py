"""
Benchmark heatmap settings on unlabeled imagery.

This script compares heatmap parameter combinations using heuristic robustness
signals such as confidence, entropy, low-confidence ratio, and spatial stability.
It does not measure true accuracy because labeled heatmap ground truth is not
available.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from stages.cell_inference.core.landcover_inference_pipeline import (
    CANONICAL_CLASS_NAMES,
    get_device,
    load_model,
    predict_heatmap,
    render_heatmap_outputs,
)


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


@dataclass(frozen=True)
class HeatmapSetting:
    cell_size: int
    overlap: float
    use_tta: bool
    temperature: float
    low_confidence_threshold: float
    neighbor_k: int
    smoothing_alpha: float


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def parse_bool_list(value: str) -> List[bool]:
    lookup = {'1': True, 'true': True, 'yes': True, 'on': True, '0': False, 'false': False, 'no': False, 'off': False}
    parsed = []
    for item in value.split(','):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized not in lookup:
            raise ValueError(f'Invalid boolean value: {item}')
        parsed.append(lookup[normalized])
    return parsed


def collect_image_paths(input_dir: str, limit: Optional[int] = None) -> List[str]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    image_paths = [str(path) for path in sorted(root.rglob('*')) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def build_settings(args) -> List[HeatmapSetting]:
    cell_sizes = parse_int_list(args.cell_sizes)
    overlaps = parse_float_list(args.overlaps)
    temperatures = parse_float_list(args.temperatures)
    low_thresholds = parse_float_list(args.low_confidence_thresholds)
    neighbor_ks = parse_int_list(args.neighbor_ks)
    smoothing_alphas = parse_float_list(args.smoothing_alphas)
    use_ttas = parse_bool_list(args.tta_values)

    combinations = itertools.product(
        cell_sizes,
        overlaps,
        use_ttas,
        temperatures,
        low_thresholds,
        neighbor_ks,
        smoothing_alphas,
    )

    return [
        HeatmapSetting(
            cell_size=cell_size,
            overlap=overlap,
            use_tta=use_tta,
            temperature=temperature,
            low_confidence_threshold=low_confidence_threshold,
            neighbor_k=neighbor_k,
            smoothing_alpha=smoothing_alpha,
        )
        for cell_size, overlap, use_tta, temperature, low_confidence_threshold, neighbor_k, smoothing_alpha in combinations
    ]


def _same_label_neighbor_ratio(grid: np.ndarray) -> float:
    if grid.size == 0:
        return 0.0

    matches = 0
    comparisons = 0
    rows, cols = grid.shape

    for row in range(rows):
        for col in range(cols):
            current = grid[row, col]
            for d_row, d_col in ((0, 1), (1, 0)):
                next_row = row + d_row
                next_col = col + d_col
                if next_row < rows and next_col < cols:
                    comparisons += 1
                    if current == grid[next_row, next_col]:
                        matches += 1

    return matches / comparisons if comparisons else 0.0


def evaluate_setting(model, device, image_paths: Sequence[str], setting: HeatmapSetting, output_dir: Optional[str], save_samples: int) -> Dict[str, object]:
    per_image: List[Dict[str, object]] = []
    label_entropy_values: List[float] = []
    confidence_values: List[float] = []
    entropy_values: List[float] = []
    low_confidence_ratios: List[float] = []
    stability_values: List[float] = []

    sample_count = 0
    sample_root = None
    if output_dir is not None:
        sample_root = os.path.join(output_dir, f'cell_{setting.cell_size}_ov_{setting.overlap}_tta_{int(setting.use_tta)}')
        os.makedirs(sample_root, exist_ok=True)

    for image_path in image_paths:
        heatmap_result = predict_heatmap(
            model,
            image_path,
            device,
            cell_size=setting.cell_size,
            overlap=setting.overlap,
            use_tta=setting.use_tta,
            temperature=setting.temperature,
            low_confidence_threshold=setting.low_confidence_threshold,
            neighbor_k=setting.neighbor_k,
            smoothing_alpha=setting.smoothing_alpha,
        )

        grid = np.array(heatmap_result['grid'], dtype=np.int64)
        confidence_grid = np.array(heatmap_result['confidence_grid'], dtype=np.float32)
        entropy_grid = np.array(heatmap_result['entropy_grid'], dtype=np.float32)

        confidence_values.append(float(heatmap_result['confidence']))
        entropy_values.append(float(np.mean(entropy_grid)))
        low_confidence_ratios.append(float(heatmap_result['low_confidence_ratio']))
        stability_values.append(_same_label_neighbor_ratio(grid))

        final_probabilities = np.array(heatmap_result['final_probabilities'], dtype=np.float32)
        label_entropy = float(-np.sum(final_probabilities * np.log(np.clip(final_probabilities, 1e-12, 1.0))))
        label_entropy_values.append(label_entropy)

        per_image.append(
            {
                'image_path': image_path,
                'predicted_class': heatmap_result['predicted_class'],
                'confidence': float(heatmap_result['confidence']),
                'grid_size': heatmap_result['grid_size'],
                'low_confidence_ratio': float(heatmap_result['low_confidence_ratio']),
                'mean_entropy': float(np.mean(entropy_grid)),
                'neighbor_stability': _same_label_neighbor_ratio(grid),
            }
        )

        if sample_root is not None and sample_count < save_samples:
            render_heatmap_outputs(image_path, heatmap_result, sample_root, prefix=Path(image_path).stem)
            sample_count += 1

    mean_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
    mean_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
    mean_low_confidence_ratio = float(np.mean(low_confidence_ratios)) if low_confidence_ratios else 0.0
    mean_neighbor_stability = float(np.mean(stability_values)) if stability_values else 0.0
    mean_label_entropy = float(np.mean(label_entropy_values)) if label_entropy_values else 0.0

    confidence_component = mean_confidence
    entropy_component = 1.0 - min(mean_entropy / max(np.log(len(CANONICAL_CLASS_NAMES)), 1e-12), 1.0)
    low_confidence_component = 1.0 - mean_low_confidence_ratio
    stability_component = mean_neighbor_stability
    collapse_penalty = min(mean_label_entropy / max(np.log(len(CANONICAL_CLASS_NAMES)), 1e-12), 1.0)

    robustness_score = (
        0.40 * confidence_component
        + 0.20 * entropy_component
        + 0.20 * low_confidence_component
        + 0.20 * stability_component
        - 0.10 * collapse_penalty
    )

    return {
        'setting': asdict(setting),
        'mean_confidence': mean_confidence,
        'mean_entropy': mean_entropy,
        'mean_low_confidence_ratio': mean_low_confidence_ratio,
        'mean_neighbor_stability': mean_neighbor_stability,
        'mean_label_entropy': mean_label_entropy,
        'robustness_score': float(robustness_score),
        'per_image': per_image,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark heatmap settings on unlabeled imagery')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images to benchmark')
    parser.add_argument('--model', type=str, default='best_eurosat_model.pth', help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='results/heatmap_benchmarks', help='Directory to store benchmark outputs')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on the number of images to evaluate')
    parser.add_argument('--save_samples', type=int, default=2, help='Number of sample heatmaps to render per setting')
    parser.add_argument('--cell_sizes', type=str, default='96,128,160', help='Comma-separated patch sizes')
    parser.add_argument('--overlaps', type=str, default='0.15,0.25,0.35', help='Comma-separated overlap values')
    parser.add_argument('--tta_values', type=str, default='true,false', help='Comma-separated booleans for TTA')
    parser.add_argument('--temperatures', type=str, default='0.9,1.0,1.2', help='Comma-separated temperature values')
    parser.add_argument('--low_confidence_thresholds', type=str, default='0.50,0.55,0.60', help='Comma-separated confidence thresholds')
    parser.add_argument('--neighbor_ks', type=str, default='4,8,12', help='Comma-separated neighbor counts')
    parser.add_argument('--smoothing_alphas', type=str, default='0.45,0.65,0.80', help='Comma-separated smoothing strengths')

    args = parser.parse_args()

    image_paths = collect_image_paths(args.image_dir, limit=args.limit)
    if not image_paths:
        raise RuntimeError(f'No images found in {args.image_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    model = load_model(args.model, device=device)

    settings = build_settings(args)
    print('=' * 80)
    print('Heatmap Benchmark')
    print('=' * 80)
    print(f'Images: {len(image_paths)}')
    print(f'Settings: {len(settings)}')
    print(f'Output dir: {os.path.abspath(args.output_dir)}')

    reports: List[Dict[str, object]] = []
    for index, setting in enumerate(settings, 1):
        print(
            f"\n[{index}/{len(settings)}] cell_size={setting.cell_size}, overlap={setting.overlap}, "
            f"tta={setting.use_tta}, temp={setting.temperature}, threshold={setting.low_confidence_threshold}, "
            f"k={setting.neighbor_k}, alpha={setting.smoothing_alpha}"
        )
        report = evaluate_setting(model, device, image_paths, setting, args.output_dir, args.save_samples)
        reports.append(report)
        print(
            f"  score={report['robustness_score']:.4f} | confidence={report['mean_confidence']:.4f} | "
            f"entropy={report['mean_entropy']:.4f} | low_conf={report['mean_low_confidence_ratio']:.4f} | "
            f"stability={report['mean_neighbor_stability']:.4f}"
        )

    sorted_reports = sorted(reports, key=lambda item: item['robustness_score'], reverse=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join(args.output_dir, f'heatmap_benchmark_{timestamp}.json')

    summary = {
        'image_dir': os.path.abspath(args.image_dir),
        'model': args.model,
        'image_count': len(image_paths),
        'setting_count': len(settings),
        'best_setting': sorted_reports[0] if sorted_reports else None,
        'reports': sorted_reports,
    }

    with open(summary_path, 'w', encoding='utf-8') as file_handle:
        json.dump(summary, file_handle, indent=2)

    print('\n' + '=' * 80)
    print('Top 5 settings by robustness score')
    print('=' * 80)
    for rank, report in enumerate(sorted_reports[:5], 1):
        setting = report['setting']
        print(
            f"{rank}. score={report['robustness_score']:.4f} | cell={setting['cell_size']} | ov={setting['overlap']} | "
            f"tta={setting['use_tta']} | temp={setting['temperature']} | threshold={setting['low_confidence_threshold']} | "
            f"k={setting['neighbor_k']} | alpha={setting['smoothing_alpha']}"
        )

    if sorted_reports:
        best = sorted_reports[0]
        best_setting = best['setting']
        print('\nBest setting:')
        print(json.dumps(best_setting, indent=2))
        print(f'Benchmark summary saved to: {summary_path}')


if __name__ == '__main__':
    main()

