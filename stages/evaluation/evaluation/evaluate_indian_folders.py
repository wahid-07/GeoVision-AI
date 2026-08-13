"""
Test Indian Satellite Data - Folder-Based Testing
Automatically reads ground truth from folder structure
Each class should be in its own folder
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import numpy as np
from pathlib import Path

from stages.landcover_model.taxonomy.landcover_class_taxonomy import CANONICAL_CLASS_NAMES, canonicalize_label
from stages.cell_inference.core.landcover_inference_pipeline import canonicalize_probabilities

CLASS_LABELS = {idx: name for idx, name in enumerate(CANONICAL_CLASS_NAMES)}
MERGED_CLASS_NAMES = CANONICAL_CLASS_NAMES
VALID_CLASSES = set(CANONICAL_CLASS_NAMES)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

# Mapping for folder names that should be treated as WaterBodies
RIVER_ALIASES = {'WaterBodies'}


def load_model(model_path, num_classes=10):
    """Load the trained Wide ResNet-50-2 model"""
    print(f"Loading model from {model_path}...")
    
    # Create model architecture (must match training architecture exactly)
    model = models.wide_resnet50_2(pretrained=False)
    n_inputs = model.fc.in_features
    
    # Check the saved model structure first
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle different model architectures
    if 'network.fc.0.weight' in state_dict:
        # Model has the full LULC_Model wrapper
        class LULC_Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = models.wide_resnet50_2(pretrained=False)
                n_inputs = self.network.fc.in_features
                self.network.fc = nn.Sequential(
                    nn.Linear(n_inputs, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes),
                    nn.LogSoftmax(dim=1)
                )
            def forward(self, xb):
                return self.network(xb)
        
        model = LULC_Model()
        model.load_state_dict(state_dict)
    elif 'fc.0.weight' in state_dict:
        # Model saved with Sequential fc layer
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
        model.load_state_dict(state_dict)
    else:
        # Simple fc layer
        model.fc = nn.Linear(n_inputs, num_classes)
        model.load_state_dict(state_dict)
    
    model.eval()
    print("âœ“ Model loaded successfully!")
    return model


def preprocess_image(image_path):
    """Load and preprocess image for prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_image


def predict(model, image_tensor, adjust_probs=True, bias_penalty=2.0, top1_penalty=1.5):
    """
    Make prediction with double penalization:
    1. Penalize dominant classes (Crop, Forest)
    2. Penalize the top-1 prediction
    If prediction survives both penalties, it's truly confident
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        
        if adjust_probs:
            # Classes to reduce: Crop (0), Forest (1)
            dominant_classes = [0, 1]
            
            # Step 1: Apply dominant class penalty
            adjusted_outputs = outputs.clone()
            for class_idx in dominant_classes:
                adjusted_outputs[0, class_idx] = adjusted_outputs[0, class_idx] - bias_penalty
            
            # Step 2: Find current top-1 prediction and penalize it
            probabilities_temp = torch.exp(adjusted_outputs)
            probabilities_temp = probabilities_temp / probabilities_temp.sum(dim=1, keepdim=True).clamp_min(1e-12)
            _, top_predicted_idx = torch.max(probabilities_temp, 1)
            
            # Apply additional penalty to top-1 prediction
            adjusted_outputs[0, top_predicted_idx] = adjusted_outputs[0, top_predicted_idx] - top1_penalty
            
            # Step 3: Final softmax after double penalization
            probabilities = torch.exp(adjusted_outputs)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            probabilities = torch.exp(outputs)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        
        merged_probs = canonicalize_probabilities(probabilities)

        confidence, predicted_idx = torch.max(merged_probs, 1)
        predicted_label = CLASS_LABELS[predicted_idx.item()]
    
    return predicted_label, confidence.item(), merged_probs[0]


def normalize_class_name(folder_name):
    """
    Normalize folder name to match merged taxonomy format
    Handles spaces, hyphens, and case variations
    """
    return canonicalize_label(folder_name)


def scan_folder_structure(test_dir):
    """
    Scan folder structure to get images organized by class
    Returns: dict mapping image paths to ground truth labels
    """
    print(f"\n{'='*80}")
    print("Scanning Folder Structure")
    print("="*80)
    
    ground_truth = {}
    class_counts = {}
    
    # Look for class folders
    for item in os.listdir(test_dir):
        item_path = os.path.join(test_dir, item)
        
        if os.path.isdir(item_path):
            # Normalize folder name to match standard class labels
            class_name = normalize_class_name(item)
            
            if class_name:
                image_count = 0
                
                # Get all images in this class folder
                for file in os.listdir(item_path):
                    file_path = os.path.join(item_path, file)
                    file_ext = Path(file).suffix.lower()
                    
                    if file_ext in IMAGE_EXTENSIONS and os.path.isfile(file_path):
                        ground_truth[file_path] = class_name
                        image_count += 1
                
                if image_count > 0:
                    class_counts[class_name] = image_count
                    print(f"  âœ“ {class_name:25s} {image_count:3d} images")
    
    total = sum(class_counts.values())
    print(f"\n  Total: {total} images across {len(class_counts)} classes")
    
    if total == 0:
        print("\n  âŒ No class folders with images found!")
        print("  Expected structure: test_dir/ClassName/image.jpg")
        return None, None
    
    return ground_truth, class_counts


def test_folder_structure(model, test_dir, save_results=True):
    """
    Test images organized in class folders
    """
    print("\n" + "="*80)
    print(f"Testing Indian Data: {test_dir}")
    print("="*80)
    
    # Scan folder structure
    ground_truth, class_counts = scan_folder_structure(test_dir)
    
    if ground_truth is None:
        return
    
    # Create output directory
    if save_results:
        output_dir = os.path.join('results', 'indian_test_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nResults will be saved to: {output_dir}")
    
    results = []
    correct = 0
    total = 0
    
    # Confusion matrix (using merged 9 classes)
    confusion = {true_class: {pred_class: 0 for pred_class in MERGED_CLASS_NAMES} 
                 for true_class in MERGED_CLASS_NAMES}
    
    # Process each image
    for idx, (img_path, true_label) in enumerate(ground_truth.items(), 1):
        img_file = os.path.basename(img_path)
        class_folder = os.path.basename(os.path.dirname(img_path))
        
        print(f"\n[{idx}/{len(ground_truth)}] {class_folder}/{img_file}")
        
        try:
            # Preprocess and predict
            image_tensor, original_image = preprocess_image(img_path)
            predicted_label, confidence, probabilities = predict(model, image_tensor)
            
            # Check correctness
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1
            total += 1
            
            # Update confusion matrix
            confusion[true_label][predicted_label] += 1
            
            # Display results
            status = "âœ“ CORRECT" if is_correct else "âœ— INCORRECT"
            print(f"  Predicted: {predicted_label} ({confidence:.2%})")
            print(f"  Ground Truth: {true_label} - {status}")
            
            # Store results
            result = {
                'filename': f"{class_folder}/{img_file}",
                'predicted_class': predicted_label,
                'confidence': float(confidence),
                'true_class': true_label,
                'correct': is_correct
            }
            results.append(result)
            
            # Save visualization (sample: first 5 per class or misclassifications)
            if save_results:
                class_dir = os.path.join(output_dir, class_folder)
                os.makedirs(class_dir, exist_ok=True)
                
                # Save all incorrect predictions and first 3 from each class
                class_result_count = sum(1 for r in results if r['true_class'] == true_label and r['filename'].startswith(class_folder))
                
                if not is_correct or class_result_count <= 3:
                    save_path = os.path.join(class_dir, f'{Path(img_file).stem}_prediction.png')
                    visualize_prediction(
                        original_image, predicted_label, confidence, probabilities,
                        true_label, is_correct, save_path
                    )
        
        except Exception as e:
            print(f"  âŒ Error processing {img_path}: {e}")
    
    # Print detailed summary
    print_summary(results, class_counts, confusion, output_dir if save_results else None)
    
    return results


def print_summary(results, class_counts, confusion, output_dir):
    """Print comprehensive test summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\nOverall Performance:")
    print(f"  Total images: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Confidence statistics
    confidences = [r['confidence'] for r in results]
    print(f"\nConfidence Statistics:")
    print(f"  Average: {np.mean(confidences):.2%}")
    print(f"  Median: {np.median(confidences):.2%}")
    print(f"  Min: {np.min(confidences):.2%}")
    print(f"  Max: {np.max(confidences):.2%}")
    
    # Per-class accuracy
    print(f"\nPer-Class Performance:")
    print(f"  {'Class':<25s} {'Total':>6s} {'Correct':>7s} {'Accuracy':>9s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*9}")
    
    class_accuracies = {}
    for class_name in sorted(class_counts.keys()):
        class_results = [r for r in results if r['true_class'] == class_name]
        class_total = len(class_results)
        class_correct = sum(1 for r in class_results if r['correct'])
        class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
        class_accuracies[class_name] = class_acc
        
        print(f"  {class_name:<25s} {class_total:>6d} {class_correct:>7d} {class_acc:>8.1f}%")
    
    # Best and worst performing classes
    if class_accuracies:
        best_class = max(class_accuracies, key=class_accuracies.get)
        worst_class = min(class_accuracies, key=class_accuracies.get)
        print(f"\n  Best performing: {best_class} ({class_accuracies[best_class]:.1f}%)")
        print(f"  Worst performing: {worst_class} ({class_accuracies[worst_class]:.1f}%)")
    
    # Common misclassifications
    print(f"\nTop 5 Misclassification Patterns:")
    misclass = []
    for true_class in confusion:
        for pred_class in confusion[true_class]:
            if true_class != pred_class and confusion[true_class][pred_class] > 0:
                misclass.append((confusion[true_class][pred_class], true_class, pred_class))
    
    misclass.sort(reverse=True)
    for count, true_c, pred_c in misclass[:5]:
        print(f"  {true_c} â†’ {pred_c}: {count} times")
    
    # Save results
    if output_dir:
        # Save JSON results
        results_file = os.path.join(output_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'test_directory': os.path.abspath(output_dir),
                'test_date': datetime.now().isoformat(),
                'total_images': total,
                'accuracy': accuracy,
                'per_class_accuracy': class_accuracies,
                'confusion_matrix': confusion,
                'results': results
            }, f, indent=2)
        print(f"\nâœ“ Detailed results saved to: {results_file}")
        
        # Save confusion matrix visualization
        plot_confusion_matrix(confusion, os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Save additional visualizations
        plot_class_accuracy(class_accuracies, os.path.join(output_dir, 'class_accuracy.png'))
        plot_confidence_distribution(results, os.path.join(output_dir, 'confidence_distribution.png'))
        plot_confidence_boxplot(results, os.path.join(output_dir, 'confidence_boxplot.png'))
        plot_prediction_distribution(results, confusion, os.path.join(output_dir, 'prediction_distribution.png'))
    
    print("="*80)


def plot_confusion_matrix(confusion, save_path):
    """Plot confusion matrix heatmap"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    classes = sorted(confusion.keys())
    matrix = np.zeros((len(classes), len(classes)))
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = confusion[true_class][pred_class]
    
    # Normalize by row (true class)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_sums, where=row_sums!=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{matrix_norm[i, j]:.2f}\n({int(matrix[i, j])})',
                          ha="center", va="center", 
                          color="white" if matrix_norm[i, j] > 0.5 else "black",
                          fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"âœ“ Confusion matrix saved to: {save_path}")


def plot_class_accuracy(class_accuracies, save_path):
    """Plot per-class accuracy bar chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    classes = sorted(class_accuracies.keys())
    accuracies = [class_accuracies[c] for c in classes]
    colors = ['green' if acc >= 70 else 'orange' if acc >= 50 else 'red' for acc in accuracies]
    
    bars = ax.bar(classes, accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.3, label='Good (70%)')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.3, label='Fair (50%)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"âœ“ Class accuracy plot saved to: {save_path}")


def plot_confidence_distribution(results, save_path):
    """Plot confidence distribution histogram for correct vs incorrect predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    correct_conf = [r['confidence'] * 100 for r in results if r['correct']]
    incorrect_conf = [r['confidence'] * 100 for r in results if not r['correct']]
    
    # Histogram for correct predictions
    ax1.hist(correct_conf, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Correct Predictions (n={len(correct_conf)})', fontsize=14, fontweight='bold', color='green')
    ax1.axvline(np.mean(correct_conf), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {np.mean(correct_conf):.1f}%')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Histogram for incorrect predictions
    ax2.hist(incorrect_conf, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Incorrect Predictions (n={len(incorrect_conf)})', fontsize=14, fontweight='bold', color='red')
    if len(incorrect_conf) > 0:
        ax2.axvline(np.mean(incorrect_conf), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(incorrect_conf):.1f}%')
        ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Confidence Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"âœ“ Confidence distribution plot saved to: {save_path}")


def plot_confidence_boxplot(results, save_path):
    """Plot confidence boxplot per class"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    classes = sorted(set(r['true_class'] for r in results))
    data_correct = []
    data_incorrect = []
    labels = []
    
    for cls in classes:
        correct_conf = [r['confidence'] * 100 for r in results if r['true_class'] == cls and r['correct']]
        incorrect_conf = [r['confidence'] * 100 for r in results if r['true_class'] == cls and not r['correct']]
        data_correct.append(correct_conf)
        data_incorrect.append(incorrect_conf)
        labels.append(cls)
    
    positions = np.arange(len(classes))
    bp1 = ax.boxplot(data_correct, positions=positions - 0.2, widths=0.35, 
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor='lightgreen', alpha=0.7),
                      medianprops=dict(color='darkgreen', linewidth=2))
    bp2 = ax.boxplot(data_incorrect, positions=positions + 0.2, widths=0.35,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor='lightcoral', alpha=0.7),
                      medianprops=dict(color='darkred', linewidth=2))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Confidence (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('True Class', fontsize=14, fontweight='bold')
    ax.set_title('Confidence Distribution by Class', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Correct', 'Incorrect'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"âœ“ Confidence boxplot saved to: {save_path}")


def plot_prediction_distribution(results, confusion, save_path):
    """Plot prediction distribution - how many predictions per class"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Count predictions per class
    pred_counts = {}
    true_counts = {}
    for r in results:
        pred_counts[r['predicted_class']] = pred_counts.get(r['predicted_class'], 0) + 1
        true_counts[r['true_class']] = true_counts.get(r['true_class'], 0) + 1
    
    # Pie chart for predicted classes
    classes_pred = list(pred_counts.keys())
    counts_pred = list(pred_counts.values())
    colors_pred = plt.cm.Set3(range(len(classes_pred)))
    
    wedges1, texts1, autotexts1 = ax1.pie(counts_pred, labels=classes_pred, autopct='%1.1f%%',
                                            colors=colors_pred, startangle=90)
    ax1.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold', pad=20)
    
    for autotext in autotexts1:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    # Pie chart for true classes
    classes_true = list(true_counts.keys())
    counts_true = list(true_counts.values())
    colors_true = plt.cm.Set3(range(len(classes_true)))
    
    wedges2, texts2, autotexts2 = ax2.pie(counts_true, labels=classes_true, autopct='%1.1f%%',
                                            colors=colors_true, startangle=90)
    ax2.set_title('True Class Distribution (Ground Truth)', fontsize=14, fontweight='bold', pad=20)
    
    for autotext in autotexts2:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"âœ“ Prediction distribution plot saved to: {save_path}")


def visualize_prediction(original_image, predicted_label, confidence, probabilities,
                        true_label=None, is_correct=None, save_path=None):
    """Visualize the prediction with confidence scores (9 merged classes)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Display original image
    ax1.imshow(original_image)
    ax1.axis('off')
    
    # Title with prediction
    title_text = f'Predicted: {predicted_label}\nConfidence: {confidence:.2%}'
    title_color = 'green'
    
    if true_label is not None:
        title_text += f'\n\nGround Truth: {true_label}'
        if is_correct:
            title_text += ' âœ“'
            title_color = 'green'
        else:
            title_text += ' âœ—'
            title_color = 'red'
    
    ax1.set_title(title_text, fontsize=14, fontweight='bold', color=title_color, pad=20)
    
    # Display confidence scores for merged classes (show only unique 6 classes)
    merged_probs_display = [
        probabilities[0].item() * 100,   # Crop
        probabilities[1].item() * 100,   # Forest
        probabilities[2].item() * 100,   # HerbaceousVegetation
        probabilities[3].item() * 100,   # Highway
        probabilities[4].item() * 100,   # Urban
        probabilities[5].item() * 100,   # WaterBodies
    ]
    
    colors = []
    for i, class_name in enumerate(MERGED_CLASS_NAMES):
        if class_name == predicted_label:
            colors.append('green')
        elif true_label and class_name == true_label:
            colors.append('orange')
        else:
            colors.append('lightblue')
    
    bars = ax2.barh(MERGED_CLASS_NAMES, merged_probs_display, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Class Probabilities (Double Penalized, 6 Classes)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    for bar, prob in zip(bars, merged_probs_display):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test model on Indian data organized in class folders'
    )
    parser.add_argument('--test_dir', type=str, 
                       default='data/IND_TESTING/COMPREHENSIVE_TEST',
                       help='Directory containing class subfolders')
    parser.add_argument('--model', type=str, 
                       default='best_eurosat_model.pth',
                       help='Path to trained model weights')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results and visualizations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_dir):
        print(f"âŒ Test directory not found: {args.test_dir}")
        print("\nCreate the folder structure with:")
        print(f"  python create_test_folders.py --output {args.test_dir}")
        return
    
    if not os.path.exists(args.model):
        print(f"âŒ Model file not found: {args.model}")
        return
    
    print("="*80)
    print("INDIAN DATA TRANSFER LEARNING TEST - FOLDER BASED")
    print("="*80)
    print(f"Model: EuroSAT Wide ResNet-50-2")
    print(f"Test Data: {args.test_dir}")
    print(f"Structure: Class folders (automatic ground truth)")
    print(f"Classes: 6 (merged taxonomy)")
    print(f"Double Penalization:")
    print(f"  1. Dominant classes (Crop, Forest): -2.0 logit penalty")
    print(f"  2. Top-1 prediction: -1.5 logit penalty")
    print(f"  â†’ If prediction survives both penalties, it's truly confident")
    print("="*80)
    
    # Load model
    model = load_model(args.model)
    
    # Test
    results = test_folder_structure(
        model, 
        args.test_dir,
        save_results=not args.no_save
    )
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

