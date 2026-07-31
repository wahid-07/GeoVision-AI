# Indian Satellite Fine-tuning Workflow

## Problem Summary
- European model (EuroSAT) achieves **<10% accuracy on Indian heatmaps**
- Root cause: Severe domain shift (different satellite characteristics, seasonal patterns, urban layouts)
- Solution: Fine-tune on Indian labeled data

## Realistic Expectations
| Labeled Data | Expected Accuracy | Time to Achieve |
|---|---|---|
| 0 (baseline) | 10% | baseline |
| 100-150 images | 55-65% | +3-4 hours labeling + 15 min fine-tune |
| 250-300 images | 70-80% | +6-8 hours labeling + 15 min fine-tune |
| 500+ images | 85-90% | +15-20 hours labeling + 15 min fine-tune |

**Key insight:** Accuracy jumps quickly after each round of new labeled data + fine-tuning.

## Phase 1: Identify Which Images to Label (Active Learning)

This step saves you time by only labeling the **most uncertain** predictions—the ones that teach the model the most.

```bash
cd c:\Users\saimj\OneDrive\Desktop\geospatial\Land-Cover-Classification-using-Sentinel-2-Dataset

# Find the 50 most uncertain predictions
python scripts/identify_uncertain_predictions.py \
  --image-dir data/IND_TESTING/COMPREHENSIVE_TEST \
  --output uncertain_images.json \
  --top-k 50
```

**Output:** `uncertain_images.json` with sorted list of most uncertain images.

**What to do:**
- Open `uncertain_images.json`
- For each image, note:
  - Full path (you'll move it)
  - Predicted class (model's guess)
  - Top 3 alternatives (help you decide if wrong)
- Manually verify what's actually in the top 50 images
- Note any images that are obviously mislabeled by the model

## Phase 2: Create Labeled Dataset Folder

Create this structure on disk:

```
labeled_indian_data/
├── Crop/
│   ├── img_001.png
│   └── img_002.png
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Urban/
├── WaterBodies/
├── Crop/
├── Urban/
└── WaterBodies/
```

**Quick way to create it:**
```bash
mkdir labeled_indian_data
for class in Crop Forest HerbaceousVegetation Highway Urban WaterBodies Crop Urban WaterBodies; do
  mkdir labeled_indian_data/$class
done
```

## Phase 3: Label Images (Your Manual Work)

**Strategy: Progressive labeling**
- **Round 1:** Label 50-100 images (prioritize high-uncertainty images from Step 1)
- **Round 2:** Label 100-150 more images
- **Round 3:** Label final 150-200 images

**For each image:**
1. Open it (it's in uncertain_images.json with path)
2. Determine the true class (what you see)
3. Move or copy to `labeled_indian_data/{TrueClass}/image.png`

**Time estimate per round:** 
- Round 1: ~2 hours (100 images @ 1 min each)
- Round 2: ~2.5 hours (easier because you've calibrated your eye)
- Round 3: ~3 hours (more complex edge cases)

## Phase 4: Fine-tune the Model (Automatic)

Once you've labeled your first batch (≥50 images), run:

```bash
python scripts/finetune_indian.py \
  --data-dir labeled_indian_data \
  --model best_eurosat_model.pth \
  --output best_indian_model.pth \
  --epochs 20 \
  --learning-rate 1e-5 \
  --batch-size 32
```

**What to expect:**
- Epoch 1-2: Accuracy jumps from 10% → 40-50%
- Epoch 5-8: Reaches 60-70%
- Epoch 12+: Plateaus at 70-80%
- **Runtime:** ~5-15 minutes (GPU: 2-3 min, CPU: 10-15 min)

**Output:**
- `best_indian_model.pth` - fine-tuned weights
- `best_indian_model_history.json` - training curves (accuracy vs epoch)

## Phase 5: Validate & Iterate

After each fine-tune round, **test your heatmap accuracy**:

```bash
python scripts/inference.py \
  --image "data/IND_TESTING/COMPREHENSIVE_TEST/WaterBodies/image.png" \
  --heatmap \
  --model best_indian_model.pth
```

**Check in output:**
- Heatmap accuracy on Indian test images
- Refinement summary (which classes are still swapping)
- Per-cell predictions in generated JSON

**If accuracy < 75% after Round 1:**
- Go back to Phase 1: re-run `identify_uncertain_predictions.py` with the new model path
- Label the next batch of hardest examples
- Repeat fine-tune

## Concrete Example: 3-Round Workflow

### Round 1: Get first 100 images labeled
```bash
# Step 1: Identify which 100 to label
python scripts/identify_uncertain_predictions.py --image-dir data/IND_TESTING/COMPREHENSIVE_TEST --top-k 100 --output uncertain_round1.json

# Step 2: Manually label (you do this - takes ~2 hours)
# Move 100 images from IND_TESTING to labeled_indian_data/{Class}/

# Step 3: Fine-tune
python scripts/finetune_indian.py --data-dir labeled_indian_data --output best_indian_model.pth --epochs 15

# Step 4: Test
python scripts/inference.py --image "test_image.png" --heatmap --model best_indian_model.pth
```

Expected result: **60-70% heatmap accuracy**

### Round 2: Add 100 more images
```bash
# Identify next hardest 100
python scripts/identify_uncertain_predictions.py --image-dir data/IND_TESTING/COMPREHENSIVE_TEST --top-k 100 --output uncertain_round2.json --model best_indian_model.pth

# Manually label new batch (~2 hours)

# Fine-tune again
python scripts/finetune_indian.py --data-dir labeled_indian_data --output best_indian_model_r2.pth --epochs 15

# Test
python scripts/inference.py --image "test_image.png" --heatmap --model best_indian_model_r2.pth
```

Expected result: **70-80% heatmap accuracy**

### Round 3: Final push to 90%
```bash
# Identify remaining uncertain examples
python scripts/identify_uncertain_predictions.py --image-dir data/IND_TESTING --top-k 150 --model best_indian_model_r2.pth

# Label final batch (~4 hours for harder examples)

# Final fine-tune
python scripts/finetune_indian.py --data-dir labeled_indian_data --output best_indian_model_final.pth --epochs 20

# Validate
python scripts/inference.py --image "test_image.png" --heatmap --model best_indian_model_final.pth
```

Expected result: **85-90% heatmap accuracy**

## FAQ

**Q: How many images do I really need?**
A: Minimum 50 per class (450 total) for decent accuracy. 100+ per class (900 total) for 90%+.

**Q: Should I label randomly or use active learning?**
A: **Always use active learning.** Labeling the 100 most-uncertain images teaches the model far more than 100 random images.

**Q: How many epochs should I fine-tune for?**
A: 15-20 epochs. Use early stopping (it's built in) — fine-tuning will stop automatically when validation loss stops improving.

**Q: What if my labeled data is biased (all same season)?**
A: Your fine-tuned model will overfit to that season. Collect images from different seasons/regions when possible.

**Q: Should I use the original model or update it each round?**
A: Update it. Use `--model best_indian_model.pth` in fine-tune script to continue from the best model so far.

## Files You Need to Create

1. **labeled_indian_data/** folder with 9 class subfolders (created manually)
2. **labeled_images** (you create by moving/copying from test set)

## Success Metrics

Check these after each fine-tune round:
- Validation accuracy in `best_indian_model_history.json` (should be 70%+)
- Heatmap test output (visual inspection)
- Terminal output from refinement summary (which class pairs are still problematic)

---

**Good luck!** The key insight: don't try to label 1000 random images. Label 100 smart ones, fine-tune, check results, then repeat.
