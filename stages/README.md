# GeoVision AI workflow structure

This folder groups the project by functional workflow stage.

## 1) Cell inference and annotation
Path: `stages/cell_inference`

### `core/`
- `single_image_inference.py` - single-image classification and optional heatmap generation
- `landcover_inference_pipeline.py` - shared model loading, preprocessing, and inference utilities

### `annotation/`
- `generate_heatmap_annotations.py` - generate cell-level predictions and annotation JSON
- `interactive_cell_annotation.py` - annotate or correct cells interactively
- `cell_annotation_dashboard.py` - web UI for reviewing cell labels and corrections
- `extract_corrected_cell_patches.py` - export corrected patches as training samples

### `active_learning/`
- `select_uncertain_predictions.py` - identify uncertain samples for manual review or retraining

## 2) Land-cover model training and serving
Path: `stages/landcover_model`

### `training/`
- `build_eurosat_csv.py` - build the training CSV from the dataset
- `train_eurosat_model.py` - train the base land-cover model
- `fine_tune_indian_model.py` - adapt the model to Indian imagery

### `taxonomy/`
- `landcover_class_taxonomy.py` - canonical class names, alias mapping, and colors

### `serving/`
- `flask_serving_app.py` - Flask API and dashboard for live predictions

## 3) Evaluation and reporting
Path: `stages/evaluation`

### `dataset_prep/`
- `create_eval_folders.py` - create labeled evaluation folder structures
- `generate_ground_truth_labels.py` - generate the ground-truth metadata

### `evaluation/`
- `evaluate_indian_folders.py` - evaluate performance on folder-based test sets
- `benchmark_heatmap_parameters.py` - benchmark heatmap settings and thresholds

### `reports/`
- `analyze_annotation_accuracy.py` - summarize model accuracy and annotation errors
