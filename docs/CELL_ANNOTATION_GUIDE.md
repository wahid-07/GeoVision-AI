# Cell-Level Annotation Workflow

## Why Cell-Level Annotation is Better

Instead of labeling entire images (1 label per image), you label individual cells (64 labels per image).

**The Math:**
- Full-image labeling: 100 images = 100 labels
- Cell-level labeling: 100 images = ~6,400 cell patches (if using 128px cells with overlap)

**This is 64x more training data from the same manual effort.**

---

## Workflow: Annotate → Extract → Train

### Step 1: Generate Annotatable Heatmap

For each image you want to label, run:

```bash
python scripts/annotate_heatmap_cells.py \
  --image path/to/satellite_image.png \
  --output-dir labeled_cells \
  --cell-size 128 \
  --overlap 0.25
```

**Output:**
- `satellite_image_heatmap_grid.png` — visualization with cell predictions
- `satellite_image_annotations.json` — annotation template (edit this)

### Step 2: Review & Mark Corrections

**Option A (Interactive - Recommended):**
```bash
python scripts/annotate_cells_interactive.py \
  --annotations satellite_image_annotations.json \
  --only-incorrect  # Only show low-confidence cells (faster)
```

The script shows you each cell one by one:
```
[Cell 1/25] 0_0
  Model prediction: Forest (45%)
  Is this correct? (y/n/skip): n
Enter correct label: WaterBodies
```

**Option B (Manual JSON Editing):**
Edit `satellite_image_annotations.json` directly. For each cell you know is wrong:
```json
{
  "cell_id": "0_0",
  "model_prediction": "Forest",
  "correct_label": "WaterBodies",
  "is_correct": false,
  "notes": "river pixel, not forest"
}
```

### Step 3: Extract Corrected Cell Patches

After marking corrections, extract the labeled cells:

```bash
python scripts/extract_corrected_cells.py \
  --annotations satellite_image_annotations.json \
  --image path/to/satellite_image.png \
  --output-dir labeled_cells
```

**Output:** organized training folder:
```
labeled_cells/
├── Forest/
│   └── satellite_image_cell_0_0.png
├── WaterBodies/
│   ├── satellite_image_cell_0_1.png
│   └── satellite_image_cell_1_2.png
├── Urban/
│   └── satellite_image_cell_2_5.png
... (all 6 merged classes)
```

### Step 4: Repeat for Multiple Images

Run Steps 1-3 on image #2, #3, etc:
```bash
# Image 2
python scripts/annotate_heatmap_cells.py --image image2.png --output-dir labeled_cells
python scripts/annotate_cells_interactive.py --annotations image2_annotations.json
python scripts/extract_corrected_cells.py --annotations image2_annotations.json --image image2.png --output-dir labeled_cells

# Image 3
python scripts/annotate_heatmap_cells.py --image image3.png --output-dir labeled_cells
# ... etc
```

**The key:** Cells from all images accumulate in `labeled_cells/{Class}/` 

### Step 5: Fine-tune on Accumulated Cell Patches

Once you've labeled 2-3 images (256-768 cell patches), fine-tune:

```bash
python scripts/finetune_indian.py \
  --data-dir labeled_cells \
  --epochs 15 \
  --output best_indian_model.pth
```

Expected accuracy jump: 10% → 60-70%

### Step 6: Test & Iterate

After fine-tuning, test the new model:

```bash
python scripts/inference.py \
  --image test_image.png \
  --heatmap \
  --model best_indian_model.pth
```

Check if accuracy improved. If not 80%+ yet:
1. Label 2-3 more images (repeat Step 1-4)
2. Fine-tune again with the new cells
3. Re-test

---

## Timeline Example

### Scenario: 10 Images with Cell Labeling

| Action | Time | Cells Generated | Total Cells |
|---|---|---|---|
| Image 1: annotate + extract | 30 min | 50 cells | 50 |
| Image 2: annotate + extract | 30 min | 45 cells | 95 |
| Image 3: annotate + extract | 30 min | 55 cells | 150 |
| **Fine-tune (Round 1)** | **15 min** | — | **150 cells** |
| **Test & evaluate** | 10 min | — | — |
| Image 4-6: annotate + extract | 90 min | 155 cells | 305 |
| **Fine-tune (Round 2)** | **15 min** | — | **305 cells** |
| Image 7-10: annotate + extract | 120 min | 210 cells | 515 |
| **Fine-tune (Round 3)** | **15 min** | — | **515 cells** |
| **Total time** | **~4.5 hours** | **515 cells** | **85-90% expected accuracy** |

**Compare to full-image labeling:**
- Full-image: 10 images = 10 labels → must label 100+ images for 90% → ~20 hours
- Cell-level: 10 images = 515 cells → 90% with ~4.5 hours

**The cell-level approach is 4x faster and gives more training data.**

---

## Best Practices for Cell Annotation

### 1. Use `--only-incorrect` Flag (Faster)

Instead of reviewing all cells, review only low-confidence ones:
```bash
python scripts/annotate_cells_interactive.py \
  --annotations image_annotations.json \
  --only-incorrect \
  --confidence-threshold 0.65
```

This shows only cells where model confidence < 65%, saving you from reviewing cells the model already got right.

### 2. Batch Similar Images

If you have 10 satellite images of the same region (same season, nearby location), label them all before fine-tuning. This teaches the model domain-specific patterns efficiently.

### 3. Prioritize High-Error Cells

After Round 1 fine-tuning, use the model to predict on new images. The cells with lowest confidence are the most informative to label next. Use `annotate_heatmap_cells.py` on these.

### 4. Document Edge Cases

For confusing cells, add notes:
```json
{
  "cell_id": "2_3",
  "model_prediction": "Highway",
  "correct_label": "WaterBodies",
  "is_correct": false,
  "notes": "Thin river/creek that looks like a road at this resolution"
}
```

These notes help you understand what the model struggles with.

---

## File Structure

After annotating 5 images, your folders look like:

```
project_root/
├── scripts/
│   ├── annotate_heatmap_cells.py       ← Step 1
│   ├── annotate_cells_interactive.py   ← Step 2
│   ├── extract_corrected_cells.py      ← Step 3
│   └── finetune_indian.py              ← Step 5
│
├── labeled_cells/                      ← Accumulated training data
│   ├── Crop/
│   │   ├── image1_cell_0_1.png
│   │   └── image3_cell_2_4.png
│   ├── Forest/
│   │   ├── image1_cell_0_0.png
│   │   ├── image2_cell_1_3.png
│   │   └── ...
│   └── WaterBodies/
│       ├── image1_cell_1_2.png
│       └── ...
│
├── image1.png
├── image1_annotations.json (after Step 2)
├── image2.png
├── image2_annotations.json
└── ... (more images)
```

---

## Command Cheat Sheet

**One image from start to finish:**
```bash
# 1. Generate heatmap and annotation template
python scripts/annotate_heatmap_cells.py --image myimage.png --output-dir labeled_cells

# 2. Interactively mark wrong cells
python scripts/annotate_cells_interactive.py --annotations myimage_annotations.json --only-incorrect

# 3. Extract corrected patches
python scripts/extract_corrected_cells.py --annotations myimage_annotations.json --image myimage.png --output-dir labeled_cells
```

**Fine-tune after labeling N images:**
```bash
python scripts/finetune_indian.py --data-dir labeled_cells --epochs 15 --output best_indian_model.pth
```

**Test the fine-tuned model:**
```bash
python scripts/inference.py --image test.png --heatmap --model best_indian_model.pth
```

---

## FAQ

**Q: How many cells do I need to reach 90% accuracy?**
A: Roughly 500-700 correctly labeled cells (8-11 images with ~60-70 cells per image). More is better but shows diminishing returns after 1000 cells.

**Q: Should I label all cells or just the wrong ones?**
A: Just the wrong ones. The `--only-incorrect` flag only shows low-confidence predictions, saving you time. Cells the model gets right don't need correction.

**Q: Can I mix cell-level and full-image labeling?**
A: Yes, both work with the `finetune_indian.py` script. Cell patches go in `labeled_cells/Class/` and full images can be in separate folders.

**Q: What if I make a labeling mistake?**
A: Re-edit the annotation JSON and re-run `extract_corrected_cells.py`. The corrected patch will overwrite the old one.

**Q: Should I label at 128px or different cell sizes?**
A: Use 128px (default). This matches what the model sees during heatmap inference, so labeled patches are directly relevant.

---

## Expected Accuracy Trajectory

| Round | Images Labeled | Cells Labeled | Model Accuracy |
|---|---|---|---|
| 0 (baseline) | 0 | 0 | 10% |
| 1 | 3 | ~180 | 55-65% |
| 2 | 6 | ~360 | 70-75% |
| 3 | 9 | ~540 | 80-85% |
| 4 | 12+ | 720+ | 85-90% |

Each 3 images you label should improve accuracy by ~5-10 percentage points.
