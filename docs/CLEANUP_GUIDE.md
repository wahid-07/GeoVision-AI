# Repository Cleanup Guide

This guide explains what files and folders can safely be removed to clean up the repository before starting the cell-level annotation workflow.

## Summary of Changes

✅ **Folder structure created**: `data/cell_training/` with organized directories for labeled cell patches
✅ **Fine-tuning script updated**: `finetune_indian.py` now defaults to `data/cell_training/labeled_cells`

## Files & Folders to Delete (Optional but Recommended)

### 1. **`data/EuroSAT_MS/`** (~1.2 GB)
- **Status**: NOT USED – only RGB imagery is used, not multispectral
- **Safe to delete**: YES
- **Command**: `rm -r data/EuroSAT_MS`

### 2. **`data/EuroSAT_RGB.zip`** (~1.0 GB)
- **Status**: Redundant compressed copy; original is in `data/EuroSAT_RGB/`
- **Safe to delete**: YES (if extraction was successful)
- **Command**: `rm data/EuroSAT_RGB.zip`
- **Note**: Verify `EuroSAT_RGB/` folder exists and contains images first

### 3. **`EuroSAT_Training_Colab.ipynb`**
- **Status**: Outdated Google Colab notebook; all training logic now in scripts
- **Safe to delete**: YES
- **Command**: `rm EuroSAT_Training_Colab.ipynb`

### 4. **`demo_result.png`**
- **Status**: Example output from earlier testing; not needed for workflow
- **Safe to delete**: YES
- **Command**: `rm demo_result.png`

### 5. **`results/`** folder (Variable size, typically 500MB+)
- **Status**: Old heatmap outputs from previous test runs; will be regenerated
- **Safe to delete**: YES
- **Command**: `rm -r results`
- **Note**: Results are date-stamped; safe to clean or keep archive

## Files & Folders to KEEP

✅ **`data/EuroSAT_RGB/`** – Training data (required)  
✅ **`data/IND_TESTING/`** – Test data (required)  
✅ **`data/reference_images/`** – Reference images (required)  
✅ **`scripts/`** – All training, inference, and cell annotation scripts  
✅ **`best_eurosat_model.pth`** – Pre-trained European model  
✅ **`app.py`** – Flask REST API  
✅ **`TESTING_GUIDE.md`** – Testing documentation  
✅ **`FINETUNING_GUIDE.md`** – Fine-tuning workflow  
✅ **`CELL_ANNOTATION_GUIDE.md`** – Cell annotation workflow  

## New Folder Structure

After cleanup, your `data/` directory should look like this:

```
data/
├── EuroSAT_RGB/              ← Pre-training data (keep)
├── IND_TESTING/              ← Indian test dataset (keep)
├── reference_images/         ← Reference images (keep)
└── cell_training/            ← NEW: For Indian cell-level training
    ├── reference_images/     ← Source images used to generate cells
    │   ├── image_001.png
    │   ├── image_001_annotations.json
    │   └── ... (accumulate over annotation rounds)
    │
    └── labeled_cells/        ← Extracted cell patches by class
        ├── Crop/
        ├── Forest/
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Urban/
        ├── WaterBodies/
        ├── Crop/
        ├── Urban/
        └── WaterBodies/
```

## Next Steps

1. **Delete the files above** (optional, but recommended)
2. **Ready to start cell annotation**:
   ```bash
   python scripts/annotate_heatmap_cells.py --image data/IND_TESTING/COMPREHENSIVE_TEST/Forest/image_001.tif
   ```
3. **Annotated cells will accumulate in** `data/cell_training/labeled_cells/{Class}/`
4. **Fine-tune without specifying path** (defaults to new location):
   ```bash
   python scripts/finetune_indian.py --epochs 15
   ```

## Cleanup Command (All At Once)

Running on Windows PowerShell:
```powershell
# Optional: Archive results before deleting
# Compress-Archive -Path results -DestinationPath results_backup.zip

# Delete unused files
Remove-Item -Recurse -Force data/EuroSAT_MS
Remove-Item data/EuroSAT_RGB.zip
Remove-Item EuroSAT_Training_Colab.ipynb
Remove-Item demo_result.png
Remove-Item -Recurse -Force results
```

Running on Linux/Mac:
```bash
# Optional: Archive results before deleting
# zip -r results_backup.zip results

# Delete unused files
rm -r data/EuroSAT_MS
rm data/EuroSAT_RGB.zip
rm EuroSAT_Training_Colab.ipynb
rm demo_result.png
rm -r results
```

## Storage Savings

- **EuroSAT_MS**: ~1.2 GB
- **EuroSAT_RGB.zip**: ~1.0 GB
- **results/**: ~0.5 GB
- **EuroSAT_Training_Colab.ipynb**: ~5 MB
- **demo_result.png**: ~1 MB
- **Total**: ~2.7 GB freed

---

**Ready to annotate?** See [CELL_ANNOTATION_GUIDE.md](CELL_ANNOTATION_GUIDE.md) for the complete workflow.
