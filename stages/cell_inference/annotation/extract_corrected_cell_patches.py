"""
Extract corrected cell patches from annotation JSON.

After you've marked corrections in the annotation JSON file,
this tool extracts the corrected cells and saves them to training folders.

Usage:
  python scripts/extract_corrected_cells.py --annotations test_image_annotations.json --image test_image.png --output-dir labeled_cells
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from PIL import Image

from stages.landcover_model.taxonomy.landcover_class_taxonomy import CANONICAL_CLASS_NAMES, canonicalize_label


def load_annotations(json_path: str) -> List[Dict]:
    """Load correction annotations from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_cell_patch(image_path: str, cell_box: List[int]) -> Image.Image:
    """Extract a cell patch from the image."""
    image = Image.open(image_path).convert('RGB')
    left, top, right, bottom = cell_box
    patch = image.crop((left, top, right, bottom))
    return patch


def main():
    parser = argparse.ArgumentParser(
        description='Extract corrected cells from annotation JSON to training folders',
        epilog='Example: python scripts/extract_corrected_cells.py --annotations test_image_annotations.json --image test_image.png --output-dir labeled_cells',
    )

    parser.add_argument('--annotations', type=str, required=True, help='Path to annotation JSON file')
    parser.add_argument('--image', type=str, required=True, help='Path to original satellite image')
    parser.add_argument('--output-dir', type=str, default='labeled_cells', help='Output directory for extracted cells')

    args = parser.parse_args()

    print('=' * 70)
    print('Extract Corrected Cell Patches')
    print('=' * 70)

    if not os.path.exists(args.annotations):
        print(f'Error: Annotation file not found: {args.annotations}')
        return

    if not os.path.exists(args.image):
        print(f'Error: Image not found: {args.image}')
        return

    print(f'\nLoading annotations from: {args.annotations}')
    annotations = load_annotations(args.annotations)
    print(f'Found {len(annotations)} cells')

    os.makedirs(args.output_dir, exist_ok=True)
    for class_name in CANONICAL_CLASS_NAMES:
        class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    print(f'\nExtractor corrected cells...')

    total_cells = len(annotations)
    corrected_cells = 0
    skipped_cells = 0

    image_stem = Path(args.image).stem

    for annotation in annotations:
        is_correct = annotation.get('is_correct', True)
        model_pred = annotation.get('model_prediction', 'Unknown')
        correct_label = canonicalize_label(annotation.get('correct_label', model_pred)) or model_pred
        cell_id = annotation.get('cell_id', 'unknown')

        if is_correct:
            skipped_cells += 1
            continue

        if correct_label not in CANONICAL_CLASS_NAMES:
            print(f'  âš ï¸  Invalid label "{correct_label}" for cell {cell_id}, skipping')
            continue

        try:
            patch = extract_cell_patch(args.image, annotation['box'])
            output_path = os.path.join(args.output_dir, correct_label, f'{image_stem}_cell_{cell_id}.png')
            patch.save(output_path)
            corrected_cells += 1
        except Exception as e:
            print(f'  Error extracting cell {cell_id}: {e}')
            continue

    print(f'\nâœ… Extraction complete!')
    print(f'  Total cells: {total_cells}')
    print(f'  Corrected cells extracted: {corrected_cells}')
    print(f'  Cells marked as correct (skipped): {skipped_cells}')
    print(f'  Output directory: {args.output_dir}/')

    print('\n' + '=' * 70)
    print('NEXT STEPS:')
    print('=' * 70)
    print('Option 1: Label more images')
    print('  - Run annotate_heatmap_cells.py on another image')
    print('  - Mark corrections')
    print('  - Run this script again (corrected cells will accumulate in labeled_cells/)')
    print('')
    print('Option 2: Fine-tune the model')
    print(f'  python scripts/finetune_indian.py --data-dir {args.output_dir} --epochs 15')
    print('=' * 70)


if __name__ == '__main__':
    main()

