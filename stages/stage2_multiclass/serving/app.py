"""
Land Cover Classification Flask API.

Provides the original single-image classifier plus a new heatmap analysis
endpoint with overlapping grids, confidence-aware refinement, and uncertainty
outputs.
"""

from datetime import datetime
from pathlib import Path
import os

from flask import Flask, jsonify, request
from PIL import Image

from stages.stage1_single_cell.core.landcover_pipeline import (
    CANONICAL_CLASS_NAMES,
    get_device,
    load_model,
    predict_heatmap,
    predict_single_image,
    render_heatmap_outputs,
)


app = Flask(__name__)

MODEL_PATH = 'best_eurosat_model.pth'
MODEL = None
DEVICE = None


def _load_image_from_upload(uploaded_file: Image.Image) -> Image.Image:
    return Image.open(uploaded_file.stream).convert('RGB')


def _serialize_top_predictions(top_predictions):
    return [
        {
            'class': item['class'],
            'confidence': f"{item['confidence']:.2%}",
            'index': item['index'],
        }
        for item in top_predictions
    ]


def initialize_model():
    global MODEL, DEVICE
    DEVICE = get_device()
    MODEL = load_model(MODEL_PATH, device=DEVICE)
    print(f'\u2713 Model loaded successfully on {DEVICE}')


@app.route('/', methods=['GET'])
def check():
    return jsonify(
        {
            'message': 'EuroSAT Land Cover Classification API',
            'status': 'active',
            'model': 'Wide ResNet-50-2',
            'classes': len(CANONICAL_CLASS_NAMES),
            'endpoints': ['/classify', '/analyze'],
        }
    )


@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image = _load_image_from_upload(uploaded_file)
        prediction = predict_single_image(MODEL, image, DEVICE, use_tta=True, temperature=1.0)

        response = {
            'predicted_class': prediction.predicted_class,
            'confidence': f"{prediction.confidence:.2%}",
            'success': True,
            'low_confidence': prediction.low_confidence,
            'top_predictions': _serialize_top_predictions(prediction.top_predictions),
        }
        return jsonify(response)

    except Exception as error:
        return jsonify({'error': str(error), 'success': False}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image = _load_image_from_upload(uploaded_file)

        cell_size = int(request.form.get('cell_size', 128))
        overlap = float(request.form.get('overlap', 0.25))
        use_tta = request.form.get('tta', 'true').lower() in {'1', 'true', 'yes', 'on'}
        temperature = float(request.form.get('temperature', 1.0))
        low_confidence_threshold = float(request.form.get('low_confidence_threshold', 0.55))
        neighbor_k = int(request.form.get('neighbor_k', 8))
        smoothing_alpha = float(request.form.get('smoothing_alpha', 0.65))
        preserve_linear_water = request.form.get('preserve_linear_water', 'true').lower() in {'1', 'true', 'yes', 'on'}
        water_preserve_threshold = float(request.form.get('water_preserve_threshold', 0.35))
        consensus_top_k = int(request.form.get('consensus_top_k', 3))
        consensus_boost = float(request.form.get('consensus_boost', 0.20))
        enable_refinement = request.form.get('enable_refinement', 'true').lower() in {'1', 'true', 'yes', 'on'}

        heatmap_result = predict_heatmap(
            MODEL,
            image,
            DEVICE,
            cell_size=cell_size,
            overlap=overlap,
            use_tta=use_tta,
            temperature=temperature,
            low_confidence_threshold=low_confidence_threshold,
            neighbor_k=neighbor_k,
            smoothing_alpha=smoothing_alpha,
            preserve_linear_water=preserve_linear_water,
            water_preserve_threshold=water_preserve_threshold,
            consensus_top_k=consensus_top_k,
            consensus_boost=consensus_boost,
            enable_refinement=enable_refinement,
        )

        file_stem = Path(uploaded_file.filename).stem or 'upload'
        output_dir = os.path.join('results', 'heatmaps', datetime.now().strftime('%Y%m%d_%H%M%S'))
        render_paths = render_heatmap_outputs(image, heatmap_result, output_dir, prefix=file_stem)

        response = {
            'success': True,
            'predicted_class': heatmap_result['predicted_class'],
            'confidence': f"{heatmap_result['confidence']:.2%}",
            'top_predictions': _serialize_top_predictions(heatmap_result['top_predictions']),
            'grid_size': heatmap_result['grid_size'],
            'cell_size': heatmap_result['cell_size'],
            'overlap': heatmap_result['overlap'],
            'low_confidence_cells': heatmap_result['low_confidence_cells'],
            'low_confidence_ratio': f"{heatmap_result['low_confidence_ratio']:.2%}",
            'artifacts': render_paths,
        }
        return jsonify(response)

    except Exception as error:
        return jsonify({'error': str(error), 'success': False}), 500


if __name__ == '__main__':
    print('=' * 60)
    print('Land Cover Classification API')
    print('=' * 60)
    initialize_model()
    print('\nStarting Flask server on http://localhost:5000')
    print("Use POST /classify for a single prediction or POST /analyze for a heatmap")
    print('=' * 60)
    app.run(debug=True, port=5000)
