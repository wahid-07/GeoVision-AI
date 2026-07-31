# Quick Testing Commands

Use either Command 1 (single image) or Command 2 (folder), then run Command 3.

```powershell
# 1) Generate plots + JSON for one image path
$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m stages.stage1_single_cell.annotation.annotate_heatmap_cells --image "data/cell_training/reference_images/image 1.png" --artifacts-root results/heatmaps/image --model best_eurosat_model.pth --skip-extraction

# 2) Generate plots + JSON for all images in a folder (new images only)
$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m stages.stage1_single_cell.annotation.annotate_heatmap_cells --image-dir "data/cell_training/reference_images" --artifacts-root results/heatmaps/image --model best_eurosat_model.pth --skip-existing --skip-extraction

# 3) Start web annotation
$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m stages.stage1_single_cell.annotation.cell_annotation_web --reference-dir "data/cell_training/reference_images" --workspace-root . --artifacts-root results/heatmaps/image --model best_eurosat_model.pth --port 8765

# 4) Start web annotation (do not auto-generate missing heatmaps)
$env:PYTHONPATH='.'; .venv\Scripts\python.exe -m stages.stage1_single_cell.annotation.cell_annotation_web --reference-dir "data/cell_training/reference_images" --workspace-root . --artifacts-root results/heatmaps/image --model best_eurosat_model.pth --port 8765 --no-auto-prepare-missing

# 5) Analyze all annotation JSON files and generate accuracy/error report plots
$env:PYTHONPATH='.'; .venv\Scripts\python.exe stages\stage3_accuracy\reports\analyze_cell_annotations.py --artifacts-root results/heatmaps/image --output-dir results
```