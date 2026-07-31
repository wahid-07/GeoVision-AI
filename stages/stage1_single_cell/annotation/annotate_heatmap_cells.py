"""
Cell-level annotation tool: Correct individual heatmap cells and export as training patches.

This tool shows the heatmap with a grid overlay, displays model predictions per cell,
and lets you mark incorrect cells with the correct label. Corrected cells are extracted
and saved as training images.

Usage:
  python scripts/annotate_heatmap_cells.py --image "path/to/image.png" --output-dir labeled_cells

The tool will:
  1. Display the heatmap with cell predictions
  2. Save an interactive JSON where you mark corrections
  3. Extract corrected cells as PNG files in labeled_cells/{Class}/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from stages.stage1_single_cell.core.landcover_pipeline import (
    CANONICAL_CLASS_COLORS,
    CANONICAL_CLASS_NAMES,
    get_device,
    load_model,
    predict_heatmap,
    render_heatmap_outputs,
)


def create_annotatable_heatmap(
    image_source,
    heatmap_result: Dict,
    output_path: str,
    cell_size: int = 128,
) -> Tuple[Image.Image, List[Dict]]:
    """
    Create a visual heatmap with cell grid and model predictions overlaid.
    Returns path to the annotatable image and a JSON template for corrections.
    """
    image = Image.open(image_source).convert('RGB')
    width, height = image.size

    heatmap_display = image.copy()
    draw = ImageDraw.Draw(heatmap_display, 'RGBA')

    grid = np.array(heatmap_result['grid'], dtype=np.int64)
    confidence_grid = np.array(heatmap_result['confidence_grid'], dtype=np.float32)
    cells = heatmap_result['cells']

    grid_height, grid_width = grid.shape
    cell_visual_width = width / grid_width
    cell_visual_height = height / grid_height

    annotation_template = []

    for row_idx in range(grid_height):
        for col_idx in range(grid_width):
            cell_idx = row_idx * grid_width + col_idx
            if cell_idx >= len(cells):
                continue

            cell_data = cells[cell_idx]
            left, top, right, bottom = cell_data['box']

            confidence = confidence_grid[row_idx, col_idx]
            predicted_class_idx = int(grid[row_idx, col_idx])
            predicted_class = CANONICAL_CLASS_NAMES[predicted_class_idx]
            color = CANONICAL_CLASS_COLORS[predicted_class]

            rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))

            draw.rectangle([left, top, right, bottom], outline=rgb + (200,), width=2)

            text = f'{predicted_class} ({confidence:.0%})'
            text_bbox = draw.textbbox((left, top), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle(
                [left, top - text_height - 4, left + text_width + 4, top],
                fill=rgb + (220,),
            )
            draw.text((left + 2, top - text_height - 2), text, fill=(255, 255, 255))

            annotation_template.append(
                {
                    'cell_id': f'{row_idx}_{col_idx}',
                    'row': row_idx,
                    'col': col_idx,
                    'box': [int(left), int(top), int(right), int(bottom)],
                    'model_prediction': predicted_class,
                    'model_confidence': float(confidence),
                    'raw_prediction': cell_data.get('raw_predicted_class', predicted_class),
                    'raw_confidence': float(cell_data.get('raw_confidence', confidence)),
                    'switched_by_refinement': bool(cell_data.get('switched_by_refinement', False)),
                    'correct_label': predicted_class,
                    'is_correct': True,
                    'notes': '',
                }
            )

    return heatmap_display, annotation_template


def extract_cell_patch(image_source, cell_box: List[int]) -> Image.Image:
    """Extract a single cell patch from the image."""
    image = Image.open(image_source).convert('RGB')
    left, top, right, bottom = cell_box
    patch = image.crop((left, top, right, bottom))
    return patch


def save_annotated_cells(
    image_source,
    annotations: List[Dict],
    output_dir: str,
) -> Tuple[int, int]:
    """
    Save corrected cell patches to training folders.
    Returns (total_cells, cells_with_corrections).
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name in CANONICAL_CLASS_NAMES:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    corrected_count = 0
    total_count = 0

    for annotation in annotations:
        if not annotation.get('is_correct', True):
            correct_label = annotation['correct_label']
            cell_box = annotation['box']
            cell_id = annotation['cell_id']

            patch = extract_cell_patch(image_source, cell_box)

            output_path = os.path.join(output_dir, correct_label, f'{Path(image_source).stem}_cell_{cell_id}.png')
            patch.save(output_path)
            corrected_count += 1

        total_count += 1

    return total_count, corrected_count


def build_artifact_paths(image_path: Path, artifacts_root: Path) -> Dict[str, Path]:
    image_stem = image_path.stem
    image_dir = artifacts_root / image_stem
    return {
        'image_dir': image_dir,
        'heatmap_grid': image_dir / f'{image_stem}_heatmap_grid.png',
        'annotations': image_dir / f'{image_stem}_annotations.json',
        'overlay': image_dir / f'{image_stem}_overlay.png',
        'confidence': image_dir / f'{image_stem}_confidence.png',
        'uncertainty': image_dir / f'{image_stem}_uncertainty.png',
        'report': image_dir / f'{image_stem}_report.png',
        'heatmap_json': image_dir / f'{image_stem}_heatmap.json',
    }


def has_complete_artifacts(paths: Dict[str, Path]) -> bool:
    required_keys = ['heatmap_grid', 'annotations', 'overlay', 'confidence', 'uncertainty', 'report', 'heatmap_json']
    return all(paths[key].exists() for key in required_keys)


def process_image(
    image_path: Path,
    model,
    device,
    artifacts_root: Path,
    output_dir: str,
    cell_size: int,
    overlap: float,
    disable_refinement: bool,
    skip_extraction: bool,
    skip_existing: bool,
) -> Dict[str, object]:
    paths = build_artifact_paths(image_path, artifacts_root)
    paths['image_dir'].mkdir(parents=True, exist_ok=True)

    if skip_existing and has_complete_artifacts(paths):
        return {'image': image_path.name, 'status': 'skipped', 'paths': paths}

    heatmap_result = predict_heatmap(
        model,
        str(image_path),
        device,
        cell_size=cell_size,
        overlap=overlap,
        use_tta=False,
        enable_refinement=not disable_refinement,
    )

    render_heatmap_outputs(str(image_path), heatmap_result, str(paths['image_dir']), prefix=image_path.stem)
    heatmap_display, annotation_template = create_annotatable_heatmap(
        str(image_path),
        heatmap_result,
        str(paths['image_dir']),
        cell_size=cell_size,
    )

    heatmap_display.save(paths['heatmap_grid'])
    with paths['annotations'].open('w', encoding='utf-8') as file_handle:
        json.dump(annotation_template, file_handle, indent=2)

    extracted_count = 0
    total_count = len(annotation_template)
    if not skip_extraction:
        total_count, extracted_count = save_annotated_cells(str(image_path), annotation_template, output_dir)

    return {
        'image': image_path.name,
        'status': 'generated',
        'paths': paths,
        'annotation_cells': len(annotation_template),
        'extracted_count': extracted_count,
        'total_count': total_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Interactive cell-level annotation tool',
        epilog='''
Example usage:
    python scripts/annotate_heatmap_cells.py --image test_image.png
    python scripts/annotate_heatmap_cells.py --image-dir data/cell_training/reference_images --skip-existing

Output:
    - results/heatmaps/image/<image_stem>/<image_stem>_annotations.json
    - results/heatmaps/image/<image_stem>/<image_stem>_overlay.png
    - results/heatmaps/image/<image_stem>/<image_stem>_confidence.png
    - results/heatmaps/image/<image_stem>/<image_stem>_uncertainty.png
    - results/heatmaps/image/<image_stem>/<image_stem>_report.png
  - labeled_cells/{Class}/ (corrected cell patches automatically extracted)
        ''',
    )

    parser.add_argument('--image', type=str, help='Path to one satellite image')
    parser.add_argument('--image-dir', type=str, help='Directory with reference images to process incrementally')
    parser.add_argument('--artifacts-root', type=str, default='results/heatmaps/image', help='Root folder where per-image heatmap artifacts are saved')
    parser.add_argument('--output-dir', type=str, default='labeled_cells', help='Output directory for labeled cell patches')
    parser.add_argument('--model', type=str, default='best_eurosat_model.pth', help='Model to use')
    parser.add_argument('--cell-size', type=int, default=128, help='Cell size for heatmap')
    parser.add_argument('--overlap', type=float, default=0.25, help='Overlap ratio')
    parser.add_argument('--disable-refinement', action='store_true', help='Use raw predictions without neighbor refinement')
    parser.add_argument('--skip-extraction', action='store_true', help='Only generate annotation JSON, skip cell extraction')
    parser.add_argument('--skip-existing', action='store_true', help='When using --image-dir, only generate artifacts for new images')

    args = parser.parse_args()

    print('=' * 70)
    print('Cell-Level Annotation Tool')
    print('=' * 70)

    if not args.image and not args.image_dir:
        print('Error: provide either --image or --image-dir')
        return

    if args.image and args.image_dir:
        print('Error: use either --image or --image-dir, not both')
        return

    artifacts_root = Path(args.artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = []
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f'Error: Image not found: {args.image}')
            return
        image_paths = [image_path]
    else:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f'Error: Image directory not found: {args.image_dir}')
            return
        valid_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        image_paths = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_suffixes], key=lambda p: p.name.lower())
        if not image_paths:
            print(f'No images found in {image_dir}')
            return

    device = get_device()
    print(f'Device: {device}\n')

    print('Loading model...')
    model = load_model(args.model, device=device)

    generated = 0
    skipped = 0
    for image_path in image_paths:
        print(f'\nProcessing: {image_path.name}')
        result = process_image(
            image_path=image_path,
            model=model,
            device=device,
            artifacts_root=artifacts_root,
            output_dir=args.output_dir,
            cell_size=args.cell_size,
            overlap=args.overlap,
            disable_refinement=args.disable_refinement,
            skip_extraction=args.skip_extraction,
            skip_existing=args.skip_existing,
        )
        if result['status'] == 'skipped':
            skipped += 1
            print('  â­ï¸  Skipped (artifacts already exist)')
            continue

        generated += 1
        result_paths = result['paths']
        print(f"  âœ… Heatmap grid: {result_paths['heatmap_grid']}")
        print(f"  âœ… Annotations: {result_paths['annotations']}")
        print(f"  âœ… Overlay: {result_paths['overlay']}")
        print(f"  âœ… Confidence: {result_paths['confidence']}")
        print(f"  âœ… Uncertainty: {result_paths['uncertainty']}")
        print(f"  âœ… Report: {result_paths['report']}")
        print(f"  âœ… JSON: {result_paths['heatmap_json']}")
        print(f"  Cells ready: {result['annotation_cells']}")

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Total images scanned: {len(image_paths)}')
    print(f'Generated: {generated}')
    print(f'Skipped existing: {skipped}')
    print(f'Artifacts root: {artifacts_root.resolve()}')
    print('Web UI will use these files when started with matching --artifacts-root.')


if __name__ == '__main__':
    main()

