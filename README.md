# Stage-Based Python File Organization

This folder groups project Python files by workflow stage.

## Stage 1: Initial Single-Cell Predictions
Path: `stages/stage1_single_cell`

### `core/`
- `inference.py` - Single image inference + optional heatmap generation
- `landcover_pipeline.py` - Shared model inference and heatmap utilities

### `annotation/`
- `annotate_heatmap_cells.py` - Generate per-cell predictions and annotation JSON
- `annotate_cells_interactive.py` - Interactive annotation support
- `cell_annotation_web.py` - Web UI for cell correction
- `extract_corrected_cells.py` - Extract corrected patches from annotation JSON

### `active_learning/`
- `identify_uncertain_predictions.py` - Select uncertain images/cells for labeling

## Stage 2: Multi-Class Classification
Path: `stages/stage2_multiclass`

### `training/`
- `create_eurosat_csv.py` - Build training CSV index from dataset
- `train.py` - Initial training on EuroSAT
- `finetune_indian.py` - Fine-tuning workflow for Indian data

### `taxonomy/`
- `class_taxonomy.py` - Canonical class mapping and label normalization

### `serving/`
- `app.py` - Flask API for model inference endpoints

## Stage 3: Accuracy and Evaluation Scripts
Path: `stages/stage3_accuracy`

### `dataset_prep/`
- `create_test_folders.py` - Build test folder structure
- `generate_ground_truth.py` - Generate ground truth metadata

### `evaluation/`
- `test_indian_data_folders.py` - Batch evaluation across labeled folders
- `benchmark_heatmap_settings.py` - Benchmark heatmap hyperparameters

### `reports/`
- `analyze_cell_annotations.py` - Annotation-driven accuracy/error analytics and plots

---

## Note
The stage folders are now the canonical source locations for these Python files.
Legacy copies under `scripts/`, `testing/`, and project root were removed during cleanup.
