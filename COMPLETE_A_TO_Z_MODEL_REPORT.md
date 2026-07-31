# Land Cover Classification Using Sentinel-2 Imagery
## Complete A-to-Z Technical Report (Training to Testing to Deployment)

Authoring date: 2026-04-26

---

## 1. Executive Summary

This project implements an end-to-end satellite land-cover classification workflow for Sentinel-2 RGB imagery, starting from baseline model training on EuroSAT and extending to adaptation, heatmap analysis, cell-level correction, active-learning sample selection, and reporting-oriented evaluation. The current implementation is organized into three stages:

- Stage 1: Initial single-cell and heatmap predictions with interactive annotation.
- Stage 2: Multi-class model training/fine-tuning and API serving.
- Stage 3: Dataset preparation, folder-based evaluation, benchmark analysis, and annotation-driven error analytics.

The project solves an important transfer-learning challenge: models trained on one domain (EuroSAT, mostly European scenes) often underperform in new geographies (Indian satellite data). To address this, the pipeline includes:

- Canonical class merging across labels.
- Confidence-aware and entropy-aware heatmap inference.
- Annotation loops for correction-driven adaptation.
- Fine-tuning on corrected data.
- Rich reporting outputs to support scientific analysis.

---

## 2. Project Objective and Research Motivation

### 2.1 Problem Statement

Given a Sentinel-2 RGB satellite image, predict the land-cover class with calibrated confidence. For larger images, generate cell-wise heatmaps and uncertainty maps to reveal spatially varying model behavior. Support domain adaptation through manual correction and iterative fine-tuning.

### 2.2 Why This Matters

Land-cover mapping underpins agriculture monitoring, urban growth analysis, flood planning, deforestation tracking, and environmental policy. A robust classification pipeline must provide:

- Image-level predictions.
- Spatial explainability.
- Uncertainty visibility.
- Error diagnostics.
- Iterative correction and retraining capability.

This project provides all five.

---

## 3. Repository Stage Architecture

The codebase was migrated into stage-wise canonical folders.

### 3.1 Stage 1: Initial Single-Cell Predictions

Path: stages/stage1_single_cell

Submodules:

- core
  - inference.py
  - landcover_pipeline.py
- annotation
  - annotate_heatmap_cells.py
  - annotate_cells_interactive.py
  - cell_annotation_web.py
  - extract_corrected_cells.py
- active_learning
  - identify_uncertain_predictions.py

### 3.2 Stage 2: Multi-Class Classification

Path: stages/stage2_multiclass

Submodules:

- training
  - create_eurosat_csv.py
  - train.py
  - finetune_indian.py
- taxonomy
  - class_taxonomy.py
- serving
  - app.py

### 3.3 Stage 3: Accuracy and Reporting

Path: stages/stage3_accuracy

Submodules:

- dataset_prep
  - create_test_folders.py
  - generate_ground_truth.py
- evaluation
  - test_indian_data_folders.py
  - benchmark_heatmap_settings.py
- reports
  - analyze_cell_annotations.py

---

## 4. Data and Label System

### 4.1 Source Dataset Concept

The project is designed around EuroSAT RGB imagery for initial training and Indian imagery for transfer testing/adaptation.

### 4.2 Canonical Label Space

The model logic uses canonical merged classes:

1. Crop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Urban
6. WaterBodies

### 4.3 Raw-to-Canonical Mapping

The taxonomy supports a raw 10-logit model output while reporting and decision-making happen in 6 canonical classes.

Raw labels include AnnualCrop, PermanentCrop, Industrial, Residential, River, SeaLake, etc., mapped as:

- AnnualCrop + PermanentCrop -> Crop
- Industrial + Residential -> Urban
- River + SeaLake (+ duplicated River index) -> WaterBodies

This design allows compatibility with checkpoints trained under legacy label indexing while preserving consistent final semantics.

---

## 5. Mathematical Formulation

### 5.1 Input and Preprocessing

Given an RGB image patch x, preprocessing is:

- Resize to 224x224.
- Convert to tensor.
- Channel normalization with ImageNet mean and std.

For channel c in {R, G, B}:

$$
\tilde{x}_c = \frac{x_c - \mu_c}{\sigma_c}
$$

where

$$
\mu = (0.485, 0.456, 0.406), \quad \sigma = (0.229, 0.224, 0.225)
$$

### 5.2 Network Output and Probabilities

The classifier head ends with LogSoftmax. For logits z:

$$
\log p_i = z_i - \log\left(\sum_j e^{z_j}\right)
$$

and probability recovery uses exponentiation plus normalization:

$$
p_i = \frac{e^{\log p_i / T}}{\sum_j e^{\log p_j / T}}
$$

where T is temperature.

### 5.3 Canonical Probability Merge

For raw probability vector p_raw and mapping M(raw_index) -> canonical_index:

$$
p_{canon}[k] = \sum_{r: M(r)=k} p_{raw}[r]
$$

Then re-normalize:

$$
p_{canon} \leftarrow \frac{p_{canon}}{\sum_k p_{canon}[k]}
$$

### 5.4 Training Objective

Because model head uses LogSoftmax, the objective is negative log-likelihood:

$$
\mathcal{L}_{NLL} = -\log p_{y}
$$

Stage-2 fine-tuning script uses CrossEntropyLoss after rebuilding the final head to canonical class count.

### 5.5 Accuracy

For N samples:

$$
\text{Accuracy} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}(\hat{y}_n = y_n)
$$

### 5.6 Entropy-Based Uncertainty

For a class-probability vector p over C classes:

$$
H(p) = -\sum_{c=1}^{C} p_c \log(p_c)
$$

Normalized entropy used in uncertainty maps:

$$
H_{norm}(p) = \frac{H(p)}{\log C}
$$

### 5.7 Active Learning Uncertainty Score

For confidence c of top class:

$$
U = 1 - c
$$

Higher U indicates better candidate for manual labeling.

### 5.8 Confidence-Aware Neighbor Refinement

For low-confidence cells, neighbor-informed smoothing uses combined distance:

$$
d_{ij} = d^{spatial}_{ij} + 0.7 \cdot d^{prob}_{ij}
$$

Neighbor weight:

$$
w_j = e^{-6 d^{spatial}_{ij}} \cdot e^{-4 d^{prob}_{ij}} \cdot conf_j
$$

Weighted neighbor average:

$$
\bar{p}_i = \frac{\sum_j w_j p_j}{\sum_j w_j}
$$

Blend with original:

$$
p'_i = (1-\alpha) p_i + \alpha \bar{p}_i
$$

Top-k consensus prior is further blended with boost beta:

$$
p''_i = (1-\beta) p'_i + \beta q_i
$$

where q_i is consensus prior restricted to top-k candidate classes.

---

## 6. Stage 2: Initial Multi-Class Training

Primary script: stages/stage2_multiclass/training/train.py

### 6.1 Model Backbone

- Wide ResNet-50-2 pretrained on ImageNet.
- Head:
  - Linear(in_features -> 256)
  - ReLU
  - Dropout(0.5)
  - Linear(256 -> num_classes)
  - LogSoftmax

### 6.2 Hyperparameters

- Batch size: 64
- Epochs: 10
- Learning rate: 1e-4
- Weight decay: 1e-3
- Gradient clipping value: 0.1
- Validation split: 0.1
- Early stopping patience: 3
- Optimizer: SGD
- LR scheduler: ReduceLROnPlateau

### 6.3 Data Loader Pipeline

Dataset built from eurosat_rgb_data.csv with image paths and labels. Labels are canonicalized using taxonomy before index conversion.

### 6.4 Validation and Model Selection

At each epoch:

- Compute train and validation losses and accuracies.
- Save checkpoint when validation accuracy improves.
- Trigger early stopping on non-improvement window.

---

## 7. Stage 2: Fine-Tuning on Indian Data

Primary script: stages/stage2_multiclass/training/finetune_indian.py

### 7.1 Goal

Adapt EuroSAT-trained representation to Indian imagery using corrected cell patches.

### 7.2 Data Source

Default data directory:

- data/cell_training/labeled_cells

Expected class-folder structure under canonical taxonomy.

### 7.3 Augmentation Strategy

Training augmentations include:

- RandomResizedCrop
- RandomRotation
- ColorJitter
- RandomHorizontalFlip

Validation transform uses deterministic resize + normalization.

### 7.4 Transfer Strategy

- Load pretrained model.
- Rebuild final classifier head to canonical class count.
- Freeze all layers then unfreeze last residual blocks + head.
- Train only trainable parameters.

### 7.5 Fine-Tuning Hyperparameters

Defaults:

- Epochs: 15
- Batch size: 32
- Learning rate: 1e-5
- Weight decay: 1e-4
- Unfreeze blocks: 2
- Validation split: 0.2
- Early stopping patience: 5
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

### 7.6 Saved Outputs

- Best model checkpoint (default best_indian_model.pth).
- Training history JSON with epoch metrics.

---

## 8. Stage 1: Single-Image Inference and Heatmaps

Primary scripts:

- stages/stage1_single_cell/core/inference.py
- stages/stage1_single_cell/core/landcover_pipeline.py

### 8.1 Single-Image Inference

Pipeline:

1. Load image.
2. Apply optional TTA (mirror, flip variants).
3. Predict probabilities.
4. Convert to canonical class probabilities.
5. Return top predictions and low-confidence flag.

### 8.2 Heatmap Generation

Large images are partitioned into overlapping cells.

Given cell size s and overlap rho, step is:

$$
\Delta = \max(1, \text{round}(s(1-\rho)))
$$

Cells are extracted over x and y grids and each cell gets probability vector, confidence, entropy, and top-k classes.

### 8.3 Refinement Logic

Refinement is enabled by default and includes:

- k-NN style spatial+probability neighbor selection.
- Confidence gating.
- WaterBodies preserve guard above threshold.
- Top-k consensus boost.
- Two-pass smoothing.

### 8.4 Rendered Heatmap Artifacts

Per image output set includes:

- overlay image
- confidence map
- uncertainty map
- composite report panel
- heatmap JSON metadata

Uncertainty decode bands are explicitly included in JSON for interpretability.

---

## 9. Stage 1: Annotation and Human-in-the-Loop Correction

Primary scripts:

- stages/stage1_single_cell/annotation/annotate_heatmap_cells.py
- stages/stage1_single_cell/annotation/cell_annotation_web.py

### 9.1 Annotation JSON Schema

Each cell record stores:

- cell_id, row, col, box
- model_prediction
- model_confidence
- raw_prediction, raw_confidence
- switched_by_refinement
- correct_label
- is_correct
- notes

This schema preserves both model output and human correction.

### 9.2 Correction Semantics

- Model output remains in model_prediction.
- Ground-truth correction is stored in correct_label.
- Error flag is stored in is_correct.

This enables confusion analysis and error hotspot mining.

### 9.3 Web Annotation API

Endpoints include:

- GET /api/images
- GET /api/annotation/<stem>
- POST /api/annotation/<stem>
- POST /api/extract/<stem>
- GET /api/file/reference/<stem>
- GET /api/file/heatmap/<stem>

### 9.4 Auto-Prepare Behavior

Web server can optionally prepare missing artifacts at startup unless disabled with no-auto-prepare-missing flag.

---

## 10. Stage 1: Active Learning

Primary script: stages/stage1_single_cell/active_learning/identify_uncertain_predictions.py

### 10.1 Workflow

- Recursively collect unlabeled images.
- Run prediction per image.
- Compute uncertainty = 1 - confidence.
- Sort descending by uncertainty.
- Export top-k uncertain samples.

### 10.2 Benefit

Prioritizes labeling budget toward high-information samples that are likely to improve model boundary quality when added to fine-tuning.

---

## 11. Stage 3: Test Dataset Preparation

Primary scripts:

- stages/stage3_accuracy/dataset_prep/create_test_folders.py
- stages/stage3_accuracy/dataset_prep/generate_ground_truth.py

### 11.1 Folder-Based Ground Truth

Ground truth is generated by folder structure and canonicalized labels.

### 11.2 Ground Truth File

Generated JSON maps relative path to class label. The script validates image extensions and class names.

---

## 12. Stage 3: Folder-Based Evaluation on Indian Data

Primary script: stages/stage3_accuracy/evaluation/test_indian_data_folders.py

### 12.1 Evaluation Flow

- Scan class folders.
- Predict each image.
- Compare predicted class vs true folder class.
- Build confusion matrix and class-wise metrics.
- Save visual diagnostics and JSON summary.

### 12.2 Double Penalization Heuristic

Evaluation script includes optional double-penalization behavior:

1. Penalize dominant classes.
2. Penalize provisional top-1.
3. Recompute probabilities.

This attempts to reduce dominant-class bias and force robust confidence.

### 12.3 Output Diagnostics

- confusion matrix image
- per-class accuracy chart
- confidence distribution histogram
- confidence boxplot
- prediction distribution chart
- detailed test_results.json

---

## 13. Stage 3: Heatmap Setting Benchmarking

Primary script: stages/stage3_accuracy/evaluation/benchmark_heatmap_settings.py

### 13.1 Purpose

Compare parameter combinations when no dense labels exist.

### 13.2 Heuristic Objective

Robustness score combines:

- mean confidence
- inverse normalized entropy
- inverse low-confidence ratio
- neighbor label stability
- collapse penalty from label entropy

Implemented form:

$$
S = 0.40 C + 0.20 E + 0.20 L + 0.20 N - 0.10 P
$$

where C, E, L, N, P represent those components.

### 13.3 Search Space

Grid over cell size, overlap, TTA, temperature, low-confidence threshold, neighbor_k, and smoothing alpha.

---

## 14. Stage 3: Annotation-Driven Error Analytics

Primary script: stages/stage3_accuracy/reports/analyze_cell_annotations.py

### 14.1 Inputs

All annotation JSON files under a heatmap artifact root.

### 14.2 Core Outputs

- summary.json
- per_class_metrics.csv
- image_error_rates.csv
- top_error_pairs.csv
- cell_level_records.csv
- multiple PNG plots
- markdown analysis report

### 14.3 Metrics

For class c:

$$
Precision_c = \frac{TP_c}{TP_c + FP_c}
$$

$$
Recall_c = \frac{TP_c}{TP_c + FN_c}
$$

$$
F1_c = \frac{2 \cdot Precision_c \cdot Recall_c}{Precision_c + Recall_c}
$$

Image-level error rate:

$$
ErrorRate_i = \frac{IncorrectCells_i}{TotalCells_i}
$$

Global annotation accuracy:

$$
Acc = \frac{\sum_i CorrectCells_i}{\sum_i TotalCells_i}
$$

---

## 15. API Deployment Layer

Primary script: stages/stage2_multiclass/serving/app.py

### 15.1 Endpoints

- GET /
  - Service metadata.
- POST /classify
  - Single-image classification.
- POST /analyze
  - Heatmap generation and artifact paths.

### 15.2 Runtime Parameters for Analyze Endpoint

Supports configurable cell size, overlap, TTA, temperature, low-confidence threshold, neighbor settings, smoothing strength, water guard, consensus options, and refinement on/off.

---

## 16. End-to-End Operational Workflow

### 16.1 Data Indexing and Baseline Training

1. Build CSV index from EuroSAT folders.
2. Train baseline model.
3. Save checkpoint.

### 16.2 Heatmap Inference and Annotation Loop

1. Generate heatmaps and annotation JSON on new images.
2. Review via web annotation UI.
3. Mark incorrect cells and corrected labels.
4. Extract corrected cell patches.

### 16.3 Active Learning Enhancement

1. Rank unlabeled images by uncertainty.
2. Label highest-uncertainty subset.
3. Add corrected cells to training pool.

### 16.4 Fine-Tuning and Re-Evaluation

1. Fine-tune on corrected Indian samples.
2. Evaluate on folder-based test set.
3. Run annotation analytics for cell-level diagnostics.

### 16.5 Reporting

Use generated JSON/CSV/plots plus this document to produce academic/project final report sections.

---

## 17. Reproducibility Commands (Stage Paths)

### 17.1 Single image heatmap generation

python -m stages.stage1_single_cell.annotation.annotate_heatmap_cells --image data/cell_training/reference_images/image 1.png --artifacts-root results/heatmaps/image --model best_eurosat_model.pth --skip-extraction

### 17.2 Batch heatmap generation for new images

python -m stages.stage1_single_cell.annotation.annotate_heatmap_cells --image-dir data/cell_training/reference_images --artifacts-root results/heatmaps/image --model best_eurosat_model.pth --skip-existing --skip-extraction

### 17.3 Web annotation

python -m stages.stage1_single_cell.annotation.cell_annotation_web --reference-dir data/cell_training/reference_images --workspace-root . --artifacts-root results/heatmaps/image --model best_eurosat_model.pth --port 8765 --no-auto-prepare-missing

### 17.4 Annotation analytics report

python stages/stage3_accuracy/reports/analyze_cell_annotations.py --artifacts-root results/heatmaps/image --output-dir results

### 17.5 Baseline training

python stages/stage2_multiclass/training/train.py

### 17.6 Fine-tuning

python stages/stage2_multiclass/training/finetune_indian.py --data-dir data/cell_training/labeled_cells --epochs 15 --output best_indian_model.pth

### 17.7 Folder-based test evaluation

python stages/stage3_accuracy/evaluation/test_indian_data_folders.py --test_dir data/IND_TESTING/COMPREHENSIVE_TEST --model best_eurosat_model.pth

### 17.8 Heatmap parameter benchmark

python stages/stage3_accuracy/evaluation/benchmark_heatmap_settings.py --image_dir data/IND_TESTING/COMPREHENSIVE_TEST --model best_eurosat_model.pth

---

## 18. Interpretation Guidance for Project Report Writing

### 18.1 Current Observed Results in This Repository Snapshot

From the latest annotation analytics run:

- Generated at: 2026-04-26T16:13:15
- Annotation files analyzed: 6
- Images analyzed: 6
- Total cells: 805
- Correct cells: 805
- Incorrect cells: 0
- Accuracy from annotation flags: 1.0000
- Error rate from annotation flags: 0.0000

Data quality checks in that run:

- rows_with_invalid_prediction: 0
- rows_with_invalid_correct_label: 0
- rows_incorrect_but_same_label: 0

Per-class support from that same run:

- Crop: 201
- Forest: 184
- HerbaceousVegetation: 14
- Highway: 13
- Urban: 161
- WaterBodies: 232

Interpretation:

This specific snapshot indicates all cells are still marked as correct in stored annotations, so the error-pair diagnostics are currently empty. As manual corrections increase, the same reporting pipeline will automatically populate confusion/error insights and misclassification hotspots.

### 18.2 If Accuracy Appears Very High in Annotation Analytics

When all cells remain marked is_correct=true, annotation-driven accuracy will be trivially 100 percent. This indicates incomplete correction pass, not necessarily perfect model quality.

### 18.3 Strong Evidence Sections for Final Thesis-Style Report

Use the following evidence stack:

1. Folder-based test metrics on unseen images.
2. Confusion matrix and class-wise performance.
3. Annotation-derived error maps and misclassification pairs.
4. Confidence versus error distribution.
5. Ablation-style benchmark of heatmap settings.

### 18.4 Recommended Narrative Flow

- Baseline model on EuroSAT.
- Domain shift to Indian data.
- Heatmap uncertainty and failure analysis.
- Human correction loop.
- Fine-tuning gains.
- Post-fine-tune error profile.

---

## 19. Potential Limitations and Risks

1. Domain gap remains if labeled Indian samples are small.
2. Class merge may hide raw subclass confusion.
3. Entropy is useful but not full epistemic uncertainty.
4. Heuristic benchmark score is not equivalent to true accuracy.
5. Annotation quality controls are needed for consistency.

---

## 20. Suggested Future Extensions

1. Calibration methods (temperature scaling learned on validation set).
2. Pixel-level segmentation for finer boundaries.
3. Geospatial metadata integration beyond RGB.
4. Semi-supervised consistency training.
5. Automated quality checks for annotation drift.

---

## 21. Mathematical Appendix

### A. Softmax

$$
softmax(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### B. LogSoftmax

$$
\log softmax(z_i) = z_i - \log\left(\sum_j e^{z_j}\right)
$$

### C. NLL Loss

$$
\mathcal{L}_{NLL} = -\log p_y
$$

### D. Entropy

$$
H(p) = -\sum_c p_c\log p_c
$$

### E. Normalized Entropy

$$
H_{norm}(p) = \frac{H(p)}{\log C}
$$

### F. Precision, Recall, F1

$$
Precision = \frac{TP}{TP+FP}
$$

$$
Recall = \frac{TP}{TP+FN}
$$

$$
F1 = \frac{2PR}{P+R}
$$

### G. Active-Learning Uncertainty

$$
U = 1 - c
$$

### H. Neighbor-Weighted Refinement

$$
\bar{p}_i = \frac{\sum_{j \in N(i)} w_j p_j}{\sum_{j \in N(i)} w_j}
$$

$$
p'_i = (1-\alpha)p_i + \alpha \bar{p}_i
$$

### I. Composite Robustness Score

$$
S = 0.40 C + 0.20 E + 0.20 L + 0.20 N - 0.10 P
$$

---

## 22. Final Conclusion

This project now represents a full lifecycle geospatial classification system:

- model development,
- uncertainty-aware spatial inference,
- human-guided correction,
- active-learning prioritization,
- transfer-learning fine-tuning,
- rich quantitative and visual reporting,
- and API deployment support.

The stage-based refactor improves maintainability and auditability, while the reporting pipeline provides a strong basis for project defense, publication-style documentation, and future extension into more advanced geospatial learning frameworks.
