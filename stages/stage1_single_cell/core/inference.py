"""
Land Cover Classification Inference Script.

Predicts land cover class from a satellite image and can optionally generate a
grid-based heatmap with confidence-aware refinement.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

from stages.stage1_single_cell.core.landcover_pipeline import (
    CANONICAL_CLASS_COLORS,
    CANONICAL_CLASS_NAMES,
    get_device,
    load_image,
    load_model,
    predict_heatmap,
    predict_single_image,
    render_heatmap_outputs,
)


def visualize_prediction(original_image, prediction, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(original_image)
    ax1.axis('off')
    ax1.set_title(
        f'Predicted: {prediction.predicted_class}\nConfidence: {prediction.confidence:.2%}',
        fontsize=14,
        fontweight='bold',
        color='green',
    )

    probabilities = prediction.probabilities.cpu().numpy()
    colors = [
        CANONICAL_CLASS_COLORS[class_name] if class_name == prediction.predicted_class else '#cbd5e1'
        for class_name in CANONICAL_CLASS_NAMES
    ]
    bars = ax2.barh(CANONICAL_CLASS_NAMES, probabilities * 100.0, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Class Probabilities (Canonical 9 Classes)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)

    for bar, prob in zip(bars, probabilities * 100.0):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{prob:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'\u2713 Visualization saved to {save_path}')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict land cover class from satellite image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='best_eurosat_model.pth', help='Path to trained model weights')
    parser.add_argument('--save', type=str, default=None, help='Path to save single-image visualization')
    parser.add_argument('--heatmap', action='store_true', help='Generate a patch-grid heatmap analysis')
    parser.add_argument('--output-dir', type=str, default='results/heatmaps', help='Directory for heatmap outputs')
    parser.add_argument('--cell-size', type=int, default=128, help='Grid cell size for heatmap inference')
    parser.add_argument('--overlap', type=float, default=0.25, help='Patch overlap ratio for heatmap inference')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation for prediction')
    parser.add_argument('--temperature', type=float, default=1.0, help='Probability temperature for the model output')
    parser.add_argument('--low-confidence-threshold', type=float, default=0.55, help='Threshold for neighbor refinement')
    parser.add_argument('--neighbor-k', type=int, default=8, help='Number of nearest neighbors for refinement')
    parser.add_argument('--smoothing-alpha', type=float, default=0.65, help='Neighbor blending weight')
    parser.add_argument('--disable-water-guard', action='store_true', help='Disable WaterBodies preservation during refinement')
    parser.add_argument('--water-preserve-threshold', type=float, default=0.35, help='Min confidence to preserve WaterBodies cells')
    parser.add_argument('--consensus-top-k', type=int, default=3, help='Number of top classes used for neighbor consensus')
    parser.add_argument('--consensus-boost', type=float, default=0.20, help='Strength of top-k consensus prior (0 to <1)')
    parser.add_argument('--disable-refinement', action='store_true', help='Skip neighbor refinement and use raw patch predictions')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f'Error: Image file not found: {args.image}')
        return

    if not os.path.exists(args.model):
        print(f'Error: Model file not found: {args.model}')
        return

    print('=' * 70)
    print('EuroSAT Land Cover Classification - Inference')
    print('=' * 70)

    device = get_device()
    model = load_model(args.model, device=device)
    image = load_image(args.image)

    print(f'\nProcessing image: {args.image}')
    print('Making prediction...')
    prediction = predict_single_image(
        model,
        image,
        device,
        use_tta=args.tta,
        temperature=args.temperature,
        low_confidence_threshold=args.low_confidence_threshold,
    )

    print('\n' + '=' * 70)
    print('PREDICTION RESULTS')
    print('=' * 70)
    print(f'Predicted Class: {prediction.predicted_class}')
    print(f'Confidence: {prediction.confidence:.2%}')
    if prediction.low_confidence:
        print('Warning: low confidence prediction, consider checking the heatmap output.')
    print('\nTop 3 Predictions:')
    for index, item in enumerate(prediction.top_predictions, 1):
        print(f"  {index}. {item['class']}: {item['confidence']:.2%}")
    print('=' * 70)

    if args.heatmap:
        print('\nGenerating heatmap...')
        heatmap_result = predict_heatmap(
            model,
            image,
            device,
            cell_size=args.cell_size,
            overlap=args.overlap,
            use_tta=args.tta,
            temperature=args.temperature,
            low_confidence_threshold=args.low_confidence_threshold,
            neighbor_k=args.neighbor_k,
            smoothing_alpha=args.smoothing_alpha,
            preserve_linear_water=not args.disable_water_guard,
            water_preserve_threshold=args.water_preserve_threshold,
            consensus_top_k=args.consensus_top_k,
            consensus_boost=args.consensus_boost,
            enable_refinement=not args.disable_refinement,
        )

        file_stem = Path(args.image).stem or 'image'
        heatmap_output_dir = os.path.join(args.output_dir, file_stem)
        render_paths = render_heatmap_outputs(image, heatmap_result, heatmap_output_dir, prefix=file_stem)

        print('Heatmap outputs:')
        for name, path in render_paths.items():
            print(f'  {name}: {path}')

        refinement_summary = heatmap_result.get('refinement_summary', {})
        if refinement_summary:
            print('\nRefinement summary:')
            print(
                f"  refined cells: {refinement_summary.get('cells_refined', 0)}/{refinement_summary.get('cells_total', 0)}"
                f" | switched class: {refinement_summary.get('cells_switched_class', 0)}"
            )
            top_switch_pairs = refinement_summary.get('top_switch_pairs', [])
            if top_switch_pairs:
                print('  top switch pairs:')
                for item in top_switch_pairs[:5]:
                    print(f"    {item['from']} -> {item['to']}: {item['count']}")

    if args.save:
        visualize_prediction(image, prediction, args.save)


if __name__ == '__main__':
    main()

