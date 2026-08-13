# GeoVision AI

This project is organized around three clear workflow areas:

## 1) Cell inference and annotation
Path: `stages/cell_inference`

### `core/`
- `single_image_inference.py` - single-image classification and optional heatmap generation
- `landcover_inference_pipeline.py` - shared model loading, preprocessing, and inference utilities

### `annotation/`
- `generate_heatmap_annotations.py` - generate cell-level model predictions and annotation JSON
- `interactive_cell_annotation.py` - interactive annotation flow for correcting predictions
- `cell_annotation_dashboard.py` - web interface for annotation review and correction
- `extract_corrected_cell_patches.py` - export corrected patches as training data

### `active_learning/`
- `select_uncertain_predictions.py` - choose uncertain samples for labeling and retraining

## 2) Land-cover model training and serving
Path: `stages/landcover_model`

### `training/`
- `build_eurosat_csv.py` - build the EuroSAT indexing CSV from the dataset
- `train_eurosat_model.py` - train the base EuroSAT classifier
- `fine_tune_indian_model.py` - refine the model on Indian imagery

### `taxonomy/`
- `landcover_class_taxonomy.py` - canonical labels, aliases, and class colors

### `serving/`
- `flask_serving_app.py` - Flask API for inference endpoints and dashboard UI

## 3) Evaluation and reporting
Path: `stages/evaluation`

### `dataset_prep/`
- `create_eval_folders.py` - build evaluation folder layouts
- `generate_ground_truth_labels.py` - generate reference labels for test datasets

### `evaluation/`
- `evaluate_indian_folders.py` - folder-based evaluation on Indian data
- `benchmark_heatmap_parameters.py` - benchmark heatmap settings and thresholds

### `reports/`
- `analyze_annotation_accuracy.py` - accuracy and annotation error analytics

---

## Project flow
- Train model in `stages/landcover_model/training`
- Run annotation and sampling in `stages/cell_inference`
- Evaluate results in `stages/evaluation`
- Serve predictions with the Flask app in `stages/landcover_model/serving`
