# Land Cover Classification using Sentinel-2 Satellite Imagery
# Complete Project Documentation

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Model Details](#architecture--model-details)
3. [Dependencies & Installation](#dependencies--installation)
4. [Project Structure](#project-structure)
5. [File Descriptions](#file-descriptions)
6. [Dataset Information](#dataset-information)
7. [Training Workflow](#training-workflow)
8. [Inference Workflow](#inference-workflow)
9. [Testing Workflow](#testing-workflow)
10. [API Usage](#api-usage)
11. [Results & Analysis](#results--analysis)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Purpose
This project implements an automated land cover classification system using deep learning on Sentinel-2 satellite imagery. The system can classify satellite images into 9 land cover types with high accuracy.

### Key Features
- ✅ **Transfer Learning**: Uses pre-trained Wide ResNet-50-2 model
- ✅ **High Accuracy**: Achieves >90% accuracy on European data
- ✅ **Cross-Domain Testing**: Evaluation on Indian satellite imagery
- ✅ **REST API**: Flask-based API for easy integration
- ✅ **Comprehensive Analytics**: Detailed performance metrics and visualizations
- ✅ **Bias Mitigation**: Novel double penalization technique for balanced predictions

### Research Context
- **Original Dataset**: EuroSAT - 27,000 labeled European satellite images
- **Publication**: Based on [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029)
- **Data Source**: European Space Agency (ESA) Sentinel-2 satellite
- **Resolution**: 10 meters per pixel
- **Spectral Bands**: 13 bands (using RGB for this project)

---

## Architecture & Model Details

### Model Architecture

**Base Model**: Wide ResNet-50-2 (Pre-trained on ImageNet)
- **Parameters**: ~68 million
- **Input**: 224×224 RGB images
- **Backbone**: 50 layers deep with wider residual blocks

**Custom Classification Head**:
```
Input (224×224×3)
    ↓
Wide ResNet-50-2 Backbone
    ↓
Flatten (2048 features)
    ↓
Linear(2048 → 256)
    ↓
ReLU Activation
    ↓
Dropout(0.5)
    ↓
Linear(256 → 10)
    ↓
LogSoftmax
    ↓
Output (10 classes)
```

**Note**: Model has 10 output neurons (WaterBodies appears at indices 5 and 8), which are merged to 9 unique classes during inference.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD with momentum (0.9) |
| Learning Rate | 1e-4 (adaptive) |
| Weight Decay | 1e-3 |
| Gradient Clipping | 0.1 |
| Batch Size | 64 |
| Epochs | 10 (with early stopping) |
| Early Stopping Patience | 3 epochs |
| Validation Split | 10% |
| Data Augmentation | Resize, Normalize (ImageNet stats) |

### Land Cover Classes (9 Unique)

1. **Crop** - Seasonal agricultural crops (wheat, rice, corn)
2. **Forest** - Dense tree coverage
3. **HerbaceousVegetation** - Grasslands, meadows, sparse vegetation
4. **Highway** - Roads, highways, paved surfaces
5. **Urban** - Factories, warehouses, industrial zones
6. **Crop** - Orchards, vineyards, plantations
7. **Urban** - Urban residential areas, houses
8. **WaterBodies** - Linear water bodies, streams
9. **WaterBodies** - Large water bodies, lakes, seas

### Bias Mitigation Strategy

**Problem**: Model over-predicts dominant classes (Crop, Forest, Crop)

**Solution**: Double Penalization Technique
1. **Step 1**: Apply -2.0 logit penalty to dominant classes
2. **Step 2**: Apply -1.5 logit penalty to the current top-1 prediction
3. **Step 3**: Recalculate softmax probabilities
4. **Result**: Predictions that survive both penalties are genuinely confident

---

## Dependencies & Installation

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB for dataset + 500MB for model
- **GPU**: Optional (CUDA-compatible GPU recommended for training)

### Required Libraries

```bash
# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0

# Visualization
matplotlib>=3.7.0

# API (if using Flask)
flask>=2.3.0

# Utilities
tqdm>=4.65.0
```

### Installation Steps

```bash
# 1. Clone or download the repository
cd Land-Cover-Classification-using-Sentinel-2-Dataset

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install torch torchvision numpy pandas pillow matplotlib flask tqdm

# 4. Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torchvision; print('TorchVision version:', torchvision.__version__)"
```

---

## Project Structure

```
Land-Cover-Classification-using-Sentinel-2-Dataset/
│
├── app.py                          # Flask REST API for classification
├── best_eurosat_model.pth          # Trained model weights (68MB)
├── eurosat_rgb_data.csv            # Dataset index file
├── TESTING_GUIDE.md                # This comprehensive guide
├── LICENCE                         # Project license
│
├── scripts/                        # Core scripts
│   ├── create_eurosat_csv.py       # Generate dataset CSV from images
│   ├── inference.py                # Single image classification + optional heatmap
│   ├── landcover_pipeline.py       # Shared inference and heatmap utilities
│   └── train.py                    # Model training script
│
├── testing/                        # Testing utilities (3 files)
│   ├── __init__.py                 # Python package marker
│   ├── create_test_folders.py      # Setup test folder structure
│   ├── generate_ground_truth.py    # Create ground truth from folders
│   └── test_indian_data_folders.py # Comprehensive testing with analytics
│
├── data/                           # Datasets
│   ├── EuroSAT_RGB/                # Training dataset (27,000 images)
│   │   ├── Crop/
│   │   ├── Forest/
│   │   ├── HerbaceousVegetation/
│   │   ├── Highway/
│   │   ├── Urban/
│   │   ├── Pasture/                # Not used in this version
│   │   ├── Crop/
│   │   ├── Urban/
│   │   ├── WaterBodies/
│   │   └── WaterBodies/
│   │
│   └── IND_TESTING/                # Test dataset (Indian images)
│       └── COMPREHENSIVE_TEST/
│           ├── Crop/
│           ├── Forest/
│           ├── HerbaceousVegetation/
│           ├── Highway/
│           ├── Urban/
│           ├── Crop/
│           ├── Urban/
│           ├── WaterBodies/
│           └── WaterBodies/
│
└── results/                        # Test results with timestamps
    └── indian_test_results_YYYYMMDD_HHMMSS/
        ├── test_results.json       # Detailed metrics
        ├── confusion_matrix.png    # Normalized confusion matrix
        ├── class_accuracy.png      # Per-class accuracy bar chart
        ├── confidence_distribution.png # Confidence histograms
        ├── confidence_boxplot.png  # Confidence by class
        ├── prediction_distribution.png # Prediction vs ground truth
        └── [ClassFolders]/         # Sample predictions with visualizations
```

### Folder Structure Explanation

#### Root Directory
- **app.py**: Flask web server providing REST API endpoints
- **best_eurosat_model.pth**: Trained model weights (PyTorch state dict)
- **eurosat_rgb_data.csv**: Index file mapping image paths to class labels

#### scripts/
Contains 3 standalone, production-ready Python scripts:
1. **create_eurosat_csv.py**: Scans EuroSAT_RGB folder and creates CSV index
2. **inference.py**: Command-line tool for single image classification and heatmaps
3. **landcover_pipeline.py**: Shared probability handling, refinement, and rendering
4. **train.py**: Complete training pipeline with validation and early stopping

#### testing/
Contains 3 files for comprehensive model evaluation:
1. **create_test_folders.py**: Creates organized folder structure for testing
2. **generate_ground_truth.py**: Generates ground truth JSON from folder names
3. **test_indian_data_folders.py**: Runs batch testing with 5 visualization plots

#### data/
Contains two main dataset categories:
1. **EuroSAT_RGB/**: Training data (27,000 European satellite images, 10 classes)
2. **IND_TESTING/**: Test data (Indian satellite images for transfer learning evaluation)

#### results/
Auto-generated timestamped folders containing:
- JSON metrics file with detailed statistics
- 5 comprehensive visualization plots
- Sample prediction images for each class

---

## File Descriptions

### Core Scripts

#### 1. scripts/train.py
**Purpose**: Complete training pipeline for the land cover classification model

**Key Features**:
- Loads EuroSAT RGB dataset from CSV
- Implements Wide ResNet-50-2 with custom classification head
- Training with SGD optimizer and learning rate scheduling
- Early stopping (patience: 3 epochs)
- Saves best model based on validation accuracy

**Usage**:
```bash
python scripts/train.py
```

**Outputs**:
- `data/model` - Trained model weights (saved as best model during training)

**Configuration** (edit in file):
- `DATA_CSV`: Path to dataset CSV
- `BATCH_SIZE`: 64 (adjust based on GPU memory)
- `EPOCHS`: 10
- `LEARNING_RATE`: 1e-4

**Training Time**:
- GPU (NVIDIA RTX 3080): ~20-30 minutes
- CPU: 2-4 hours

---

#### 2. scripts/inference.py
**Purpose**: Classify a single satellite image and visualize results

**Key Features**:
- Loads trained model
- Preprocesses input image (resize, normalize)
- Predicts land cover class with confidence score
- Displays top-3 predictions
- Generates visualization with probability bar chart

**Usage**:
```bash
# Basic prediction
python scripts/inference.py --image path/to/image.jpg

# Save visualization
python scripts/inference.py --image path/to/image.jpg --save output.png

# Use custom model
python scripts/inference.py --image path/to/image.jpg --model my_model.pth

# Generate heatmap analysis
python scripts/inference.py --image path/to/image.jpg --heatmap --output-dir results/heatmaps

# Tune grid and refinement settings
python scripts/inference.py --image path/to/image.jpg --heatmap --cell-size 128 --overlap 0.25 --tta
```

**Outputs**:
- Console: Predicted class, confidence, top-3 predictions
- Visualization: Original image + probability bar chart

---

#### 3. scripts/create_eurosat_csv.py
**Purpose**: Scan EuroSAT_RGB dataset and create CSV index file

**Key Features**:
- Recursively scans class folders
- Finds all image files (.jpg, .png, .tif)
- Creates shuffled CSV with image paths and labels
- Prints dataset statistics

**Usage**:
```bash
python scripts/create_eurosat_csv.py
```

**Outputs**:
- `eurosat_rgb_data.csv` - Dataset index file

**When to Use**:
- When you first download the EuroSAT dataset
- When you add/remove images from the dataset
- When reorganizing dataset structure

---

### Testing Utilities

#### 4. testing/create_test_folders.py
**Purpose**: Create organized folder structure for testing

**Usage**:
```bash
python testing/create_test_folders.py --output data/IND_TESTING/NEW_TEST
```

**Outputs**:
- Creates 9 class folders
- Generates README.txt with instructions

---

#### 5. testing/generate_ground_truth.py
**Purpose**: Automatically generate ground truth JSON from folder structure

**Key Features**:
- Scans class folders
- Maps image paths to class labels
- Validates image extensions
- Prints dataset balance statistics

**Usage**:
```bash
python testing/generate_ground_truth.py --input data/IND_TESTING/COMPREHENSIVE_TEST
```

**Outputs**:
- `ground_truth.json` in the input directory

---

#### 6. testing/test_indian_data_folders.py
**Purpose**: Comprehensive batch testing with detailed analytics

**Key Features**:
- Automatic ground truth from folder structure
- Batch prediction on all test images
- Confusion matrix calculation
- Per-class accuracy analysis
- 5 comprehensive visualizations
- JSON export of all results

**Usage**:
```bash
# Basic testing
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/COMPREHENSIVE_TEST

# Custom model
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/COMPREHENSIVE_TEST --model my_model.pth

# Don't save results
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/COMPREHENSIVE_TEST --no_save
```

**Outputs** (in `results/indian_test_results_TIMESTAMP/`):
1. **test_results.json** - Complete metrics and results
2. **confusion_matrix.png** - 9×9 normalized confusion matrix
3. **class_accuracy.png** - Bar chart of per-class accuracy
4. **confidence_distribution.png** - Histograms (correct vs incorrect)
5. **confidence_boxplot.png** - Confidence distribution by class
6. **prediction_distribution.png** - Predicted vs true class distribution

**Metrics Provided**:
- Overall accuracy
- Per-class accuracy, precision, recall
- Confidence statistics (mean, median, min, max)
- Top-5 misclassification patterns
- Detailed prediction log for every image

---

### API and Main Application

#### 7. app.py
**Purpose**: Flask REST API for web-based classification

**Endpoints**:

##### GET /
Health check endpoint
```bash
curl http://localhost:5000/
```
Response:
```json
{
  "message": "EuroSAT Land Cover Classification API",
  "status": "active",
  "model": "Wide ResNet-50-2",
  "classes": 9
}
```

##### POST /classify
Classify uploaded image
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/classify
```
Response:
```json
{
  "predicted_class": "Forest",
  "confidence": "92.45%",
  "success": true
}
```

**Usage**:
```bash
# Start server
python app.py

# Server runs on http://localhost:5000
```

**Integration Examples**:

Python:
```python
import requests

url = "http://localhost:5000/classify"
files = {'file': open('satellite_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

JavaScript:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/classify', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## Dataset Information

### EuroSAT RGB Dataset

**Source**: [EuroSAT GitHub Repository](https://github.com/phelber/eurosat)

**Specifications**:
- **Total Images**: 27,000
- **Image Size**: 64×64 pixels (resized to 224×224 for training)
- **Format**: JPEG/PNG
- **Geographic Coverage**: 13 European countries
- **Satellite**: Sentinel-2, Level-1C
- **Resolution**: 10 meters per pixel
- **Time Period**: 2015-2017
- **Cloud Coverage**: <10% (pre-filtered)

**Class Distribution**:
| Class | Images | Percentage |
|-------|--------|------------|
| Crop | 3,000 | 11.1% |
| Forest | 3,000 | 11.1% |
| HerbaceousVegetation | 3,000 | 11.1% |
| Highway | 2,500 | 9.3% |
| Urban | 2,500 | 9.3% |
| Pasture | 2,000 | 7.4% |
| Crop | 2,500 | 9.3% |
| Urban | 3,000 | 11.1% |
| WaterBodies | 2,500 | 9.3% |
| WaterBodies | 3,000 | 11.1% |

**Download Instructions**:
```bash
# Option 1: Download from GitHub (recommended)
git clone https://github.com/phelber/eurosat.git
cd eurosat
# Extract EuroSAT_RGB.zip to data/EuroSAT_RGB/

# Option 2: Direct download
# Visit: https://github.com/phelber/eurosat
# Download EuroSAT_RGB.zip
# Extract to: data/EuroSAT_RGB/EuroSAT_RGB/
```

### Indian Test Dataset

**Purpose**: Evaluate transfer learning performance on different geographic region

**Collection Method**:
1. Source: Sentinel Hub EO Browser + ISRO Bhuvan
2. Manual selection of cloud-free images
3. Regions: Bihar, Punjab, Kerala, Maharashtra, Gujarat
4. Time Period: 2023-2024
5. Manual labeling by class

**Test Dataset Specifications**:
- **Total Images**: 107
- **Classes**: 9 (excluding Pasture)
- **Image Size**: Variable (resized to 224×224 for inference)
- **Format**: JPEG/PNG
- **Average per Class**: ~12 images

**Why Different Accuracy?**
- **Domain Shift**: European vs Indian geography
- **Climate Zones**: Different vegetation patterns
- **Agricultural Practices**: Different crop types and cycles
- **Urban Design**: Different residential and industrial patterns
- **Expected Accuracy Drop**: 20-30% (observed: ~30%)

---

## Training Workflow

### Step 1: Prepare Dataset

```bash
# 1. Download EuroSAT RGB dataset
# Extract to: data/EuroSAT_RGB/EuroSAT_RGB/

# 2. Verify folder structure
# Should see: Crop/, Forest/, etc.

# 3. Generate CSV index
python scripts/create_eurosat_csv.py
```

**Expected Output**:
```
Creating EuroSAT RGB Dataset CSV
Scanning directory: data/EuroSAT_RGB/EuroSAT_RGB
Crop               3000 images
Forest                   3000 images
...
Total images: 27000
CSV saved to: eurosat_rgb_data.csv
```

### Step 2: Train Model

```bash
python scripts/train.py
```

**Training Output**:
```
EuroSAT Land Cover Classification Training
Using device: cuda
Loading dataset from: eurosat_rgb_data.csv
Total images: 27000
Training samples: 24300
Validation samples: 2700

Creating model...
Model created (Wide ResNet-50-2 with transfer learning)

Starting Training
Epoch [1/10]
  Train Loss: 0.3245 | Train Acc: 89.23%
  Val Loss: 0.2156 | Val Acc: 92.45%
  New best model saved!

Epoch [2/10]
  Train Loss: 0.1987 | Train Acc: 93.12%
  Val Loss: 0.1876 | Val Acc: 94.23%
  New best model saved!

...

Training Complete!
Best Val Acc: 94.67%
Model saved to: data/model
```

**Note**: Training time varies:
- **GPU (RTX 3080)**: 20-30 minutes
- **GPU (GTX 1060)**: 45-60 minutes
- **CPU**: 2-4 hours

### Step 3: Verify Model

```bash
# Test on a sample image
python scripts/inference.py --image data/EuroSAT_RGB/EuroSAT_RGB/Forest/Forest_1.jpg
```

**Expected Output**:
```
Loading model from best_eurosat_model.pth...
✓ Model loaded successfully!

Processing image: data/EuroSAT_RGB/EuroSAT_RGB/Forest/Forest_1.jpg
Making prediction...

PREDICTION RESULTS
Predicted Class: Forest
Confidence: 95.67%

Top 3 Predictions:
  1. Forest: 95.67%
  2. Crop: 3.21%
  3. Crop: 0.89%
```

---

## Inference Workflow

### Single Image Classification

```bash
# Basic usage
python scripts/inference.py --image path/to/satellite_image.jpg

# Save visualization
python scripts/inference.py --image path/to/image.jpg --save result.png

# Use custom model
python scripts/inference.py --image path/to/image.jpg --model my_model.pth
```

### Batch Processing (Python Script)

Create a custom script:
```python
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import glob

# Load model (same as inference.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.wide_resnet50_2(pretrained=False)
# ... (load weights)

# Get all images
image_paths = glob.glob('data/test_images/*.jpg')

# Process each
for img_path in image_paths:
    # Preprocess and predict
    # ... (same as inference.py)
    print(f"{img_path}: {predicted_class} ({confidence:.2%})")
```

### Heatmap Benchmarking

When you do not have labels, use the benchmark script to compare heatmap settings on the same image set.

```bash
# Run a parameter sweep over unlabeled imagery
python testing/benchmark_heatmap_settings.py --image_dir data/IND_TESTING/COMPREHENSIVE_TEST --model best_eurosat_model.pth

# Limit the number of images and save sample overlays for each setting
python testing/benchmark_heatmap_settings.py --image_dir data/IND_TESTING/COMPREHENSIVE_TEST --limit 20 --save_samples 2

# Narrow the search space if you want a faster sweep
python testing/benchmark_heatmap_settings.py --image_dir data/IND_TESTING/COMPREHENSIVE_TEST --cell_sizes 96,128 --overlaps 0.2,0.3 --neighbor_ks 4,8
```

The benchmark does not compute true accuracy because labels are unavailable. It ranks settings using a robustness score built from confidence, entropy, low-confidence ratio, and spatial stability. Use the top settings as candidates, then inspect the rendered overlays before locking the defaults.

### Using API

```python
import requests

# Start Flask server first: python app.py

# Send image
url = "http://localhost:5000/classify"
files = {'file': open('satellite_image.jpg', 'rb')}
response = requests.post(url, files=files)

result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}")
```

---

## Testing Workflow

### Step 1: Prepare Test Data

```bash
# 1. Create test folder structure
python testing/create_test_folders.py --output data/IND_TESTING/MY_TEST

# 2. Manually copy labeled images into class folders
# Copy crop images to: data/IND_TESTING/MY_TEST/Crop/
# Copy forest images to: data/IND_TESTING/MY_TEST/Forest/
# ... etc

# 3. Verify structure
# Each class folder should contain 10-50 images
```

### Step 2: Generate Ground Truth (Optional)

```bash
# Auto-generate from folder structure
python testing/generate_ground_truth.py --input data/IND_TESTING/MY_TEST
```

### Step 3: Run Comprehensive Test

```bash
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/MY_TEST
```

### Step 4: Analyze Results

**Results Location**: `results/indian_test_results_TIMESTAMP/`

**Files Generated**:
1. **test_results.json** - All metrics and predictions
2. **confusion_matrix.png** - See which classes are confused
3. **class_accuracy.png** - Identify best/worst performing classes
4. **confidence_distribution.png** - Understand model confidence
5. **confidence_boxplot.png** - Compare confidence across classes
6. **prediction_distribution.png** - Check for bias

**Sample Output**:
```
COMPREHENSIVE TEST SUMMARY

Overall Performance:
  Total images: 107
  Correct predictions: 64
  Accuracy: 59.81%

Per-Class Performance:
  Class                     Total Correct  Accuracy
  -----------------------------------------------
  WaterBodies                      10       9    90.0%
  Urban                  13      11    84.6%
  Forest                       12      10    83.3%
  Crop                10       8    80.0%
  Urban                   11       8    72.7%
  WaterBodies                        11       5    45.5%
  HerbaceousVegetation          9       4    44.4%
  Highway                      15       3    20.0%
  Crop                   15       2    13.3%

Top 5 Misclassification Patterns:
  Crop → Crop: 7 times
  Highway → WaterBodies: 5 times
  HerbaceousVegetation → Crop: 4 times
  WaterBodies → Highway: 3 times
  Urban → Urban: 3 times
```

---

## API Usage

### Starting the Server

```bash
python app.py
```

**Output**:
```
Land Cover Classification API
✓ Model loaded successfully on cuda

Starting Flask server on http://localhost:5000
Use POST /classify for a single prediction or POST /analyze for a heatmap
```

### API Endpoints

#### 1. Health Check - GET /

**Request**:
```bash
curl http://localhost:5000/
```

**Response**:
```json
{
  "message": "EuroSAT Land Cover Classification API",
  "status": "active",
  "model": "Wide ResNet-50-2",
  "classes": 9
}
```

#### 2. Classify Image - POST /classify

**Request** (curl):
```bash
curl -X POST -F "file=@satellite_image.jpg" http://localhost:5000/classify
```

**Request** (Python):
```python
import requests

url = "http://localhost:5000/classify"
files = {'file': open('satellite_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Success Response**:
```json
{
  "predicted_class": "Forest",
  "confidence": "92.45%",
  "low_confidence": false,
  "top_predictions": [
    {"class": "Forest", "confidence": "92.45%", "index": 1},
    {"class": "Urban", "confidence": "3.10%", "index": 7},
    {"class": "Urban", "confidence": "1.74%", "index": 4}
  ],
  "success": true
}
```

#### 3. Heatmap Analysis - POST /analyze

**Request** (curl):
```bash
curl -X POST -F "file=@satellite_image.jpg" http://localhost:5000/analyze
```

**Success Response**:
```json
{
  "success": true,
  "predicted_class": "Forest",
  "confidence": "81.13%",
  "grid_size": [4, 5],
  "cell_size": 128,
  "overlap": 0.25,
  "low_confidence_cells": 3,
  "low_confidence_ratio": "15.00%",
  "artifacts": {
    "heatmap_path": "results/heatmaps/..._heatmap.png",
    "overlay_path": "results/heatmaps/..._overlay.png",
    "confidence_path": "results/heatmaps/..._confidence.png",
    "uncertainty_path": "results/heatmaps/..._uncertainty.png",
    "report_path": "results/heatmaps/..._report.png",
    "json_path": "results/heatmaps/..._heatmap.json"
  }
}
```

**Error Response**:
```json
{
  "error": "No file provided",
  "success": false
}
```

### Integration Examples

#### Python Client
```python
import requests
import glob

api_url = "http://localhost:5000/classify"

# Classify multiple images
for img_path in glob.glob('test_images/*.jpg'):
    with open(img_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(api_url, files=files)
        result = response.json()
        
        if result['success']:
            print(f"{img_path}: {result['predicted_class']} ({result['confidence']})")
        else:
            print(f"{img_path}: Error - {result['error']}")
```

#### JavaScript (Web Application)
```javascript
async function classifyImage(fileInput) {
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch('http://localhost:5000/classify', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log(`Predicted: ${result.predicted_class}`);
            console.log(`Confidence: ${result.confidence}`);
        } else {
            console.error(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error('API request failed:', error);
    }
}
```

---

## Results & Analysis

### Training Results (European Data)

**Dataset**: 27,000 EuroSAT RGB images
**Split**: 90% training, 10% validation
**Final Performance**:
- **Validation Accuracy**: 94.67%
- **Training Time**: ~25 minutes (RTX 3080)
- **Best Epoch**: 8/10
- **Early Stopping**: Triggered after epoch 10

**Per-Class Accuracy** (European Validation):
| Class | Accuracy |
|-------|----------|
| WaterBodies | 98.2% |
| Urban | 96.5% |
| Forest | 95.8% |
| Urban | 94.3% |
| Highway | 93.7% |
| Crop | 93.1% |
| Crop | 92.4% |
| WaterBodies | 91.8% |
| HerbaceousVegetation | 90.5% |

### Transfer Learning Results (Indian Data)

**Dataset**: 107 manually labeled Indian satellite images
**Performance**:
- **Overall Accuracy**: 59.81%
- **Accuracy Drop**: ~35% (expected for cross-domain transfer)
- **High Performers**: WaterBodies (90%), Urban (84.6%), Forest (83.3%)
- **Low Performers**: Crop (13.3%), Highway (20.0%)

**Key Findings**:
1. **Natural features transfer better** (water, forests)

### Testing Scripts

**Folder-based testing:**
```bash
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/COMPREHENSIVE_TEST
```
Automatically uses folder structure as ground truth. No JSON file needed.

**Single image prediction:**
```bash
python scripts/inference.py --image path/to/image.jpg --save output.png
```

### Utility Scripts

**Create folder structure:**
```bash
python testing/create_test_folders.py --output data/IND_TESTING/NEW_TEST
```

**Generate ground truth JSON:**
```bash
python testing/generate_ground_truth.py --input data/IND_TESTING/COMPREHENSIVE_TEST
```
Scans folders and creates `ground_truth.json` automatically.

---

## Understanding Results

### Overall Metrics

```
Overall Performance:
  Total images: 500
  Correct predictions: 387
  Accuracy: 77.40%

Confidence Statistics:
  Average: 68.45%
  Median: 71.20%
  Min: 32.15%
  Max: 98.76%
```

### Per-Class Performance

```
Per-Class Accuracy:
  WaterBodies:              96.0% (Best - distinctive features)
  Forest:               96.0%
  WaterBodies:                90.0%
  Urban:          86.0%
  Highway:              80.0%
  Crop:           84.0%
  Urban:           76.0%
  HerbaceousVegetation: 70.0%
  Crop:        72.0%
```

### Common Misclassifications

- **Crop ↔ Crop** - Similar vegetation patterns
- **HerbaceousVegetation ↔ Crop** - Both are grasslands
- **Highway ↔ Urban** - Linear man-made structures

---

## Model Architecture

### Architecture Details

**Base Model**: Wide ResNet-50-2 (pretrained on ImageNet)

**Custom Classification Head**:
```
Linear(2048 → 256) → ReLU → Dropout(0.5) → Linear(256 → 9) → LogSoftmax
```

**Training**:
- Dataset: EuroSAT RGB (27,000 European satellite images)
- Epochs: 10 with early stopping
- Optimizer: SGD with learning rate scheduling
- Input size: 224×224 RGB images

**9 Land Cover Classes**:
1. Crop - Seasonal agricultural fields
2. Forest - Dense tree coverage
3. HerbaceousVegetation - Grasslands, meadows, shrublands
4. Highway - Major roads and highways
5. Urban - Factories, industrial areas
6. Crop - Orchards, vineyards, plantations
7. Urban - Housing, urban residential areas
8. WaterBodies - WaterBodiess and streams
9. WaterBodies - Lakes, seas, water bodies

---

## Transfer Learning Performance

### Why Test on Indian Data?

The model was trained on **European satellite imagery** but we're testing on **Indian satellite imagery**. This evaluates how well learned features transfer across geographic regions.

### Expected Accuracy by Class

| Class Type | Expected Accuracy | Reason |
|------------|-------------------|--------|
| Water bodies (WaterBodies, WaterBodies) | 85-95% | Distinctive spectral signatures |
| Dense vegetation (Forest) | 85-95% | Clear patterns |
| Urban features (Urban, Highway) | 75-85% | Good structural features |
| Agricultural land (crops) | 60-75% | Regional variations |

**Overall Expected**: 65-80% accuracy without fine-tuning

### Factors Affecting Accuracy

**Domain Shift**:
- Different climate and vegetation types
- Indian landscapes vs European landscapes
- Seasonal differences

**Good Transfer**:
- Low-level features (edges, textures) transfer well
- Water bodies easily recognized

**Poor Transfer**:
- Agricultural land types vary by region
- Building/infrastructure styles differ

---

## Improving Accuracy

### Option 1: Use As-Is (70-80% accuracy)

If current accuracy is acceptable for your use case, no changes needed.

### Option 2: Fine-tune on Indian Data

**Collect labeled Indian data:**
- 1000-2000 labeled Indian satellite images
- Organize in same folder structure
- Mix of all 10 classes

**Fine-tune command:**
```bash
python scripts/train.py --pretrained best_eurosat_model.pth --data indian_train.csv --epochs 5
```

**Expected improvement**: 80-95% accuracy after fine-tuning

### Option 3: Domain Adaptation

Use advanced techniques like domain adversarial training (requires code modifications).

---

## Image Requirements

### Format
- **Supported**: .jpg, .jpeg, .png, .tif, .tiff
- **Recommended**: .jpg for efficiency

### Quality
- Clear satellite/aerial view
- RGB color images (3 channels)
- Minimal cloud coverage
- Any size (auto-resized to 224×224)

### Where to Get Indian Satellite Images

1. **Sentinel Hub** - https://www.sentinel-hub.com/
2. **Bhuvan (ISRO)** - https://bhuvan.nrsc.gov.in/
3. **Google Earth Engine** - https://earthengine.google.com/
4. **EO Browser** - https://apps.sentinel-hub.com/eo-browser/

---

## File Structure

### Project Structure
```
app.py                      # Flask REST API (root level)

src/                        # Core model code
├── model.py                # Model architecture
├── config.py               # Configuration
├── dataset.py              # Dataset utilities
└── predict.py              # Prediction utilities

scripts/                    # Training & inference scripts
├── train.py                # Training script
├── inference.py            # Single image prediction
└── create_eurosat_csv.py   # Dataset preparation

testing/                    # Testing tools
├── test_indian_data_folders.py  # Main test script
├── create_test_folders.py       # Create folder structure
└── generate_ground_truth.py     # Generate labels

data/
├── EuroSAT_RGB/           # Training data
├── IND_TESTING/           # Test data
└── best_eurosat_model.pth # Trained model
```

### Data Structure
```
data/
├── EuroSAT_RGB/           # Training data (European)
└── IND_TESTING/           # Your test data (Indian)
    ├── BIHAR/             # Example: 2 test images
    └── COMPREHENSIVE_TEST/ # Main test set (500 images)
        ├── Crop/
        ├── Forest/
        └── ... (10 classes)
```

---

## API Usage

### Start Flask Server
```bash
python app.py
```

### Test API
```bash
# Check status
curl http://localhost:5000/

# Classify image
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/classify
```

**Response**:
```json
{
  "Type": "Forest"
}
```

---

## Troubleshooting

### Issue: No images found
**Cause**: Images not in class folders  
**Solution**: Ensure images are directly in class folders (e.g., `Crop/image.jpg`)

### Issue: Low accuracy (<60%)
**Causes**:
1. Incorrect ground truth labels
2. Poor image quality
3. Significant domain shift

**Solutions**:
1. Verify labels are correct
2. Use clear, cloud-free images
3. Consider fine-tuning on Indian data

### Issue: Model file not found
**Cause**: Missing `best_eurosat_model.pth`  
**Solution**: Train model first with `python train.py`

### Issue: Slow processing
**Causes**: Large images, many images  
**Solutions**:
- Use smaller image files (<2MB)
- Test in batches
- Use `--no_save` flag to skip visualizations

---

## Example Workflow

### Small Test (10 images)
```bash
# Create test folder
mkdir data/IND_TESTING/SMALL_TEST
mkdir data/IND_TESTING/SMALL_TEST/Forest

# Add 10 forest images
# Copy images to data/IND_TESTING/SMALL_TEST/Forest/

# Test
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/SMALL_TEST
```

### Full Test (500 images)
```bash
# Already created: data/IND_TESTING/COMPREHENSIVE_TEST/

# Add 50 images per class to appropriate folders

# Test
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/COMPREHENSIVE_TEST

# Results saved to: results/indian_test_results_YYYYMMDD_HHMMSS/
```

### Regional Comparison
```bash
# Test North India
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/NORTH_INDIA

# Test South India
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/SOUTH_INDIA

# Compare results
```

---

## Key Commands Reference

### Setup
```bash
# Create folder structure
python testing/create_test_folders.py --output data/IND_TESTING/COMPREHENSIVE_TEST

# Generate ground truth JSON (optional)
python testing/generate_ground_truth.py --input data/IND_TESTING/COMPREHENSIVE_TEST
```

### Testing
```bash
# Test with folder structure
python testing/test_indian_data_folders.py --test_dir data/IND_TESTING/COMPREHENSIVE_TEST

# Single image
python scripts/inference.py --image path/to/image.jpg
```

### Training
```bash
# Train from scratch
python scripts/train.py

# Fine-tune on Indian data
python scripts/train.py --pretrained best_eurosat_model.pth --data indian_data.csv
```

### API
```bash
# Start server
python app.py

# Test endpoint
curl -X POST -F "file=@image.jpg" http://localhost:5000/classify
```

---

## Summary

**What this project does:**
- Classifies satellite images into 10 land cover types
- Uses transfer learning with Wide ResNet-50-2
- Trained on European data (EuroSAT)
- Can test on Indian data to evaluate transfer learning

**Testing workflow:**
1. Organize images by class in folders
2. Run `python test_indian_data_folders.py --test_dir <folder>`
3. Review results in `results/` folder
4. Analyze per-class performance

**Expected results:**
- 65-80% accuracy (transfer learning)
- Water and forest classes perform best (85-95%)
- Agricultural classes more challenging (60-75%)
- Fine-tuning can improve to 85-95%

**For more details:**
- Original project: See [README.md](README.md)
- Single image inference: See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)

---

**Last Updated**: March 9, 2026  
**Model**: Wide ResNet-50-2  
**Training Data**: EuroSAT (European satellite imagery)  
**Test Data**: Indian satellite imagery (transfer learning)
