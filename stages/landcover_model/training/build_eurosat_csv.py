"""
Create CSV file for EuroSAT RGB Dataset
This script scans the EuroSAT_RGB folder and creates a CSV file
listing all images with their labels.
"""

import os
import pandas as pd
from pathlib import Path

from stages.landcover_model.taxonomy.landcover_class_taxonomy import CANONICAL_CLASS_NAMES, CLASS_ALIASES

# Configuration
DATA_DIR = 'data/EuroSAT_RGB/EuroSAT_RGB'  # Root directory with class folders
OUTPUT_CSV = 'eurosat_rgb_data.csv'

# Expected merged classes and their legacy folder aliases
EXPECTED_CLASSES = CANONICAL_CLASS_NAMES

# Valid image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

def create_dataset_csv(data_dir, output_csv):
    """
    Scan image folders and create CSV file
    
    Args:
        data_dir: Root directory containing class folders
        output_csv: Output CSV filename
    """
    
    print("=" * 60)
    print("ðŸ“‚ Creating EuroSAT RGB Dataset CSV")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"\nâŒ Error: Data directory not found: {data_dir}")
        print("\nPlease check your folder structure.")
        return None
    
    data = []
    class_counts = {}
    
    print(f"\nScanning directory: {data_dir}")
    print("-" * 60)
    
    # Scan each merged class, looking in legacy source folders too
    for class_name in EXPECTED_CLASSES:
        candidate_folders = [class_name] + CLASS_ALIASES.get(class_name, [])
        class_dirs = [os.path.join(data_dir, folder) for folder in candidate_folders if os.path.exists(os.path.join(data_dir, folder))]

        if not class_dirs:
            print(f"âš ï¸  Warning: Folder not found: {class_name}/")
            class_counts[class_name] = 0
            continue
        
        # Find all images in this folder
        image_files = []
        for class_dir in class_dirs:
            for ext in IMAGE_EXTENSIONS:
                image_files.extend(Path(class_dir).glob(f'*{ext}'))
                image_files.extend(Path(class_dir).glob(f'*{ext.upper()}'))
        
        count = 0
        for img_path in image_files:
            # Store relative path from project root
            relative_path = os.path.relpath(str(img_path), data_dir)
            data.append({
                'image_path': relative_path,
                'label': class_name
            })
            count += 1
        
        class_counts[class_name] = count
        
        if count > 0:
            print(f"âœ… {class_name:25s} {count:5d} images")
        else:
            print(f"âš ï¸  {class_name:25s} {count:5d} images (EMPTY!)")
    
    print("-" * 60)
    
    if not data:
        print("\nâŒ Error: No images found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nâœ… Dataset CSV created: {output_csv}")
    print(f"Total images: {len(df)}")
    
    # Print statistics
    print("\nðŸ“Š Class Distribution:")
    print("-" * 60)
    for class_name in EXPECTED_CLASSES:
        count = class_counts.get(class_name, 0)
        percentage = (count / len(df) * 100) if len(df) > 0 else 0
        print(f"{class_name:25s} {count:5d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("âœ… CSV file ready for training!")
    print("=" * 60)
    
    return df

def main():
    """Main function"""
    df = create_dataset_csv(DATA_DIR, OUTPUT_CSV)
    
    if df is not None:
        # Show first few rows
        print("\nFirst 10 rows of dataset:")
        print(df.head(10))
        print(f"\nCSV saved to: {OUTPUT_CSV}")
        print("You can now use this CSV for training.")

if __name__ == '__main__':
    main()

