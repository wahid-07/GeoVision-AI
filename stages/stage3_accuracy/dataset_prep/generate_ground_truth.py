"""
Generate Ground Truth JSON from Folder Structure
Automatically creates ground_truth.json by scanning class folders
"""

import os
import json
import argparse
from pathlib import Path

from stages.stage2_multiclass.taxonomy.class_taxonomy import CANONICAL_CLASS_NAMES, CLASS_ALIASES, canonicalize_label

# Valid image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

VALID_CLASSES = set(CANONICAL_CLASS_NAMES)


def generate_ground_truth(input_dir, output_file=None):
    """
    Generate ground truth JSON from folder structure
    
    Args:
        input_dir: Directory containing class subfolders
        output_file: Output JSON file path (default: input_dir/ground_truth.json)
    """
    
    print("="*70)
    print("Generating Ground Truth from Folder Structure")
    print("="*70)
    print(f"\nScanning directory: {input_dir}\n")
    
    if not os.path.exists(input_dir):
        print(f"âŒ Error: Directory not found: {input_dir}")
        return
    
    ground_truth = {}
    stats = {}
    
    # Scan each subdirectory
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        
        # Check if it's a directory and a valid class
        canonical_class = canonicalize_label(item)
        if os.path.isdir(item_path) and canonical_class in VALID_CLASSES:
            class_name = canonical_class
            image_count = 0
            
            # Get all image files in this class folder
            for file in os.listdir(item_path):
                file_ext = Path(file).suffix.lower()
                
                if file_ext in IMAGE_EXTENSIONS:
                    # Store as: "ClassName/image.jpg" -> "ClassName"
                    relative_path = f"{class_name}/{file}"
                    ground_truth[relative_path] = class_name
                    image_count += 1
            
            if image_count > 0:
                stats[class_name] = image_count
                print(f"  âœ“ {class_name:25s} {image_count:3d} images")
            else:
                print(f"  âš  {class_name:25s} 0 images (folder empty)")
    
    # Summary
    total_images = sum(stats.values())
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Total classes found: {len(stats)}/10")
    print(f"Total images: {total_images}")
    
    if total_images == 0:
        print("\nâŒ No images found! Please add images to class folders.")
        return
    
    # Calculate balance
    if stats:
        avg_per_class = total_images / len(stats)
        print(f"Average per class: {avg_per_class:.1f}")
        
        min_class = min(stats, key=stats.get)
        max_class = max(stats, key=stats.get)
        print(f"Min images: {stats[min_class]} ({min_class})")
        print(f"Max images: {stats[max_class]} ({max_class})")
        
        # Check balance
        if stats[max_class] > 2 * stats[min_class]:
            print("\nâš ï¸  Warning: Dataset is imbalanced!")
            print("   Consider adding more images to classes with fewer samples.")
    
    # Save ground truth JSON
    if output_file is None:
        output_file = os.path.join(input_dir, 'ground_truth.json')
    
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\nâœ“ Ground truth saved to: {output_file}")
    
    # Show next steps
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("Run the test with:")
    print(f"  python test_indian_data.py --test_dir {input_dir} --ground_truth {output_file}")
    print("\nOr use the folder-based test (automatically uses folder structure):")
    print(f"  python test_indian_data_folders.py --test_dir {input_dir}")
    print("="*70)
    
    return ground_truth, stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate ground truth JSON from folder structure'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing class folders')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: input_dir/ground_truth.json)')
    parser.add_argument('--show_files', action='store_true',
                       help='Show all individual file mappings')
    
    args = parser.parse_args()
    
    ground_truth, stats = generate_ground_truth(args.input, args.output)
    
    # Optionally show all file mappings
    if args.show_files and ground_truth:
        print("\n" + "="*70)
        print("File Mappings (sample)")
        print("="*70)
        count = 0
        for filename, class_label in sorted(ground_truth.items()):
            print(f"  {filename:50s} â†’ {class_label}")
            count += 1
            if count >= 20:  # Show first 20
                remaining = len(ground_truth) - count
                if remaining > 0:
                    print(f"  ... and {remaining} more files")
                break


if __name__ == '__main__':
    main()

