"""
Interactive CLI annotation tool: Quickly mark incorrect cells.

Shows one cell at a time with model prediction and lets you type corrections.

Usage:
  python scripts/annotate_cells_interactive.py --annotations test_image_annotations.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from stages.stage2_multiclass.taxonomy.class_taxonomy import CANONICAL_CLASS_NAMES, canonicalize_label


def display_cell_for_annotation(annotation: Dict, cell_number: int, total_cells: int) -> Dict:
    """
    Display a cell annotation and ask user for correction if needed.
    Returns the updated annotation.
    """
    cell_id = annotation['cell_id']
    model_pred = annotation['model_prediction']
    model_conf = annotation['model_confidence']

    print(f'\n[Cell {cell_number}/{total_cells}] {cell_id}')
    print(f'  Model prediction: {model_pred} ({model_conf:.0%})')
    print(f'  Valid classes: {", ".join(CANONICAL_CLASS_NAMES)}')
    print()

    response = input('Is this correct? (y/n/skip): ').strip().lower()

    if response == 'y' or response == '':
        annotation['is_correct'] = True
        annotation['correct_label'] = model_pred
        return annotation

    if response == 'skip':
        return annotation

    correct_label = input('Enter correct label: ').strip()
    correct_label = canonicalize_label(correct_label) or correct_label
    if correct_label:
        annotation['is_correct'] = False
        annotation['correct_label'] = correct_label
        annotation['notes'] = f'Corrected from {model_pred}'
    else:
        annotation['is_correct'] = True

    return annotation


def main():
    parser = argparse.ArgumentParser(
        description='Interactive cell annotation tool',
        epilog='Example: python scripts/annotate_cells_interactive.py --annotations test_image_annotations.json',
    )

    parser.add_argument('--annotations', type=str, required=True, help='Path to annotation JSON file')
    parser.add_argument('--only-incorrect', action='store_true', help='Only show low-confidence cells')
    parser.add_argument('--confidence-threshold', type=float, default=0.70, help='Show cells below this confidence')

    args = parser.parse_args()

    if not os.path.exists(args.annotations):
        print(f'Error: File not found: {args.annotations}')
        return

    print('=' * 70)
    print('Interactive Cell Annotation')
    print('=' * 70)

    with open(args.annotations, 'r') as f:
        annotations = json.load(f)

    if args.only_incorrect:
        before_filter = len(annotations)
        annotations = [
            a for a in annotations if a['model_confidence'] < args.confidence_threshold
        ]
        print(f'Filtered to {len(annotations)} low-confidence cells (< {args.confidence_threshold:.0%})')
    else:
        print(f'Total cells to review: {len(annotations)}')

    annotated_count = 0
    for idx, annotation in enumerate(annotations):
        annotation = display_cell_for_annotation(annotation, idx + 1, len(annotations))
        annotations[idx] = annotation
        annotated_count += 1

        if (idx + 1) % 5 == 0:
            print(f'\nâœ“ Annotated {idx + 1}/{len(annotations)} cells')

    with open(args.annotations, 'w') as f:
        json.dump(annotations, f, indent=2)

    corrected_count = sum(1 for a in annotations if not a.get('is_correct', True))
    print('\n' + '=' * 70)
    print(f'âœ… Annotation complete!')
    print(f'  Marked as incorrect: {corrected_count}')
    print(f'  Marked as correct: {len(annotations) - corrected_count}')
    print(f'  Updated file: {args.annotations}')
    print('')
    print('Next: python scripts/extract_corrected_cells.py --annotations "{}" --image "your_image.png" --output-dir labeled_cells'.format(
        args.annotations
    ))
    print('=' * 70)


if __name__ == '__main__':
    main()

