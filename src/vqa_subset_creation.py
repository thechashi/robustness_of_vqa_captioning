import json
import os
import shutil
from pathlib import Path

def create_vqa_subset_folder():
    # Define paths
    source_folder = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/datasets/coco/val2014"
    subset_json = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/vqa_subset_20_per_type.json"
    
    # Create subset folder next to source folder
    subset_folder = os.path.join(os.path.dirname(source_folder), "vqa_subset_images")
    os.makedirs(subset_folder, exist_ok=True)
    
    # Load the subset JSON file
    with open(subset_json, 'r') as f:
        subset_data = json.load(f)
    
    # Keep track of statistics
    total_images = 0
    copied_images = 0
    missing_images = []
    
    # Create set of unique image filenames
    unique_images = set()
    for sample in subset_data['samples']:
        unique_images.add(sample['image_filename'])
    
    print(f"Found {len(unique_images)} unique images in subset JSON")
    
    # Copy each image
    for image_filename in unique_images:
        source_path = os.path.join(source_folder, image_filename)
        dest_path = os.path.join(subset_folder, image_filename)
        
        total_images += 1
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            copied_images += 1
        else:
            missing_images.append(image_filename)
    
    # Print summary
    print(f"\nSubset Creation Summary:")
    print(f"Subset folder created at: {subset_folder}")
    print(f"Total images to copy: {total_images}")
    print(f"Successfully copied: {copied_images}")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print("\nMissing image files:")
        for img in missing_images:
            print(f"  - {img}")
    
    return subset_folder

if __name__ == "__main__":
    subset_folder = create_vqa_subset_folder()