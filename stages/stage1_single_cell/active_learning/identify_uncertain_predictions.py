"""
Active Learning: Identify the most uncertain predictions.

This helps you label the IMAGE with the HIGHEST information value first.
Run inference on unlabeled images, then label ONLY the lowest-confidence ones.

Usage:
  python scripts/identify_uncertain_predictions.py --image-dir data/IND_TESTING/COMPREHENSIVE_TEST --output uncertain_images.json --top-k 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch

from stages.stage1_single_cell.core.landcover_pipeline import get_device, load_model, predict_single_image


def collect_image_paths(input_dir: str, extensions: set = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}) -> List[str]:
    """Recursively collect image paths."""
    root = Path(input_dir)
    image_paths = []
    for ext in extensions:
        image_paths.extend([str(p) for p in root.rglob(f'*{ext}') if p.is_file()])
        image_paths.extend([str(p) for p in root.rglob(f'*{ext.upper()}') if p.is_file()])
    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(
        description='Find most uncertain predictions (active learning)',
        epilog='Example: python scripts/identify_uncertain_predictions.py --image-dir data/IND_TESTING/COMPREHENSIVE_TEST --output uncertain.json',
    )

    parser.add_argument('--image-dir', type=str, required=True, help='Directory with unlabeled images (can be nested)')
    parser.add_argument('--model', type=str, default='best_eurosat_model.pth', help='Model to use for inference')
    parser.add_argument('--output', type=str, default='uncertain_images.json', help='Output JSON file')
    parser.add_argument('--top-k', type=int, default=20, help='Number of most uncertain images to report')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size if processing in parallel')

    args = parser.parse_args()

    print('=' * 70)
    print('Active Learning: Identify Uncertain Predictions')
    print('=' * 70)

    device = get_device()
    print(f'Device: {device}\n')

    print('Loading model...')
    model = load_model(args.model, device=device)

    print('Collecting images...')
    image_paths = collect_image_paths(args.image_dir)
    print(f'Found {len(image_paths)} images\n')

    if not image_paths:
        print('No images found!')
        return

    print('Running inference (this may take a while)...\n')

    results: List[Dict] = []

    for idx, image_path in enumerate(image_paths):
        try:
            prediction = predict_single_image(model, image_path, device, use_tta=False)
            confidence = prediction.confidence
            predicted_class = prediction.predicted_class
            top_3 = prediction.top_predictions

            # Uncertainty score: lower confidence = higher uncertainty
            uncertainty = 1.0 - confidence

            results.append(
                {
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': float(confidence),
                    'uncertainty': float(uncertainty),
                    'top_3_predictions': [
                        {'class': item['class'], 'confidence': float(item['confidence'])} for item in top_3
                    ],
                }
            )

            if (idx + 1) % 10 == 0:
                print(f'  Processed {idx + 1}/{len(image_paths)} images')

        except Exception as e:
            print(f'Error processing {image_path}: {e}')
            continue

    # Sort by uncertainty (highest uncertainty first)
    results.sort(key=lambda x: x['uncertainty'], reverse=True)

    top_uncertain = results[: args.top_k]

    print(f'\n' + '=' * 70)
    print(f'Top {args.top_k} MOST UNCERTAIN predictions')
    print('=' * 70)
    print(f'{"Path":50s} | {"Pred":18s} | {"Conf":6s} | Top-2 Alternatives')
    print('-' * 70)

    for item in top_uncertain:
        path = Path(item['image_path']).name
        pred_class = item['predicted_class'][:16]
        conf = f"{item['confidence']:.1%}"
        alt1, alt2 = item['top_3_predictions'][1:3]
        alt_text = f"{alt1['class'][:12]} ({alt1['confidence']:.0%}), {alt2['class'][:12]} ({alt2['confidence']:.0%})"

        print(f'{path:50s} | {pred_class:18s} | {conf:6s} | {alt_text}')

    print('=' * 70)

    output_data = {
        'total_images': len(results),
        'top_uncertain_count': len(top_uncertain),
        'sorted_by_uncertainty': top_uncertain,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f'\nâœ… Results saved to {args.output}')
    print('\nNEXT STEPS:')
    print(f'1. Label the top {args.top_k} images (highest uncertainty = most informative)')
    print(f'2. Move them to folders: labeled_indian_data/ClassName/image.png')
    print(f'3. Run: python scripts/finetune_indian.py --data-dir labeled_indian_data --epochs 15')
    print('=' * 70)


if __name__ == '__main__':
    main()

