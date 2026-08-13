"""
Create Test Folder Structure
Automatically creates organized folders for testing with 10 land cover classes
"""

import os
import argparse

from stages.landcover_model.taxonomy.landcover_class_taxonomy import CANONICAL_CLASS_NAMES

CLASSES = CANONICAL_CLASS_NAMES


def create_test_folders(output_dir):
    """Create folder structure for organized testing"""
    
    print("="*60)
    print("Creating Test Folder Structure")
    print("="*60)
    
    # Create main directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"âœ“ Created main directory: {output_dir}")
    else:
        print(f"â„¹ Main directory already exists: {output_dir}")
    
    # Create class folders
    print("\nCreating class folders:")
    for class_name in CLASSES:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"  âœ“ {class_name}/")
        else:
            print(f"  â„¹ {class_name}/ (already exists)")
    
    # Create a README file in the directory
    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("Indian Satellite Image Test Dataset\n")
        f.write("="*50 + "\n\n")
        f.write("Folder Structure:\n")
        f.write("Place your labeled images in the appropriate class folders.\n\n")
        f.write("Classes:\n")
        for i, class_name in enumerate(CLASSES, 1):
            f.write(f"  {i:2d}. {class_name}/\n")
        f.write("\nRecommended: ~50 images per class\n")
        f.write("Total target: ~500 images\n\n")
        f.write("After adding images, run:\n")
        f.write(f"  python generate_ground_truth.py --input {output_dir}\n")
        f.write("\nThen test with:\n")
        f.write(f"  python test_indian_data.py --test_dir {output_dir}\n")
    
    print(f"\nâœ“ Created README: {readme_path}")
    
    print("\n" + "="*60)
    print("Folder Structure Created Successfully!")
    print("="*60)
    print(f"\nLocation: {os.path.abspath(output_dir)}")
    print(f"\nNext steps:")
    print("  1. Copy your labeled images into the appropriate class folders")
    print(f"     Example: Copy crop images to {output_dir}/Crop/")
    print(f"  2. Run: python generate_ground_truth.py --input {output_dir}")
    print(f"  3. Run: python test_indian_data.py --test_dir {output_dir}")
    print("\nRecommended: ~50 images per class for comprehensive testing")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Create organized test folder structure')
    parser.add_argument('--output', type=str, 
                       default='data/IND_TESTING/COMPREHENSIVE_TEST',
                       help='Output directory for test folders')
    
    args = parser.parse_args()
    create_test_folders(args.output)


if __name__ == '__main__':
    main()

