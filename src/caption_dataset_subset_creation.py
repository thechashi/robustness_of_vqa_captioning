import json
import random
from collections import defaultdict
from pathlib import Path
import os
import shutil

def create_caption_subset(captions_file, images_dir, n_samples=5, output_file="caption_subset.json", copy_images=True):
    """
    Create a subset of COCO caption dataset with n different images and their captions.
    
    Args:
        captions_file: Path to COCO captions file (e.g., captions_val2014.json)
        images_dir: Directory containing the images
        n_samples: Number of images to sample
        output_file: Output JSON file path
        copy_images: Whether to copy selected images to a new directory
    """
    # Load caption data
    print(f"Loading captions from {captions_file}")
    with open(captions_file, 'r') as f:
        data = json.load(f)
    
    # Create image lookup
    image_lookup = {img['id']: img for img in data['images']}
    
    # Group captions by image_id
    captions_by_image = defaultdict(list)
    for ann in data['annotations']:
        captions_by_image[ann['image_id']].append(ann)
    
    # Get all available image IDs
    available_images = list(captions_by_image.keys())
    print(f"Total available images: {len(available_images)}")
    
    # Randomly sample n images
    selected_image_ids = random.sample(available_images, min(n_samples, len(available_images)))
    
    # Create subset
    subset_images = []
    subset_annotations = []
    image_dir_name = "subset_images"
    
    if copy_images:
        # Create directory for copied images
        subset_image_dir = os.path.join(os.path.dirname(output_file), image_dir_name)
        os.makedirs(subset_image_dir, exist_ok=True)
    
    for image_id in selected_image_ids:
        image_info = image_lookup[image_id]
        image_filename = image_info['file_name']
        image_path = os.path.join(images_dir, image_filename)
        
        # Only include if image exists
        if os.path.exists(image_path):
            # Add image info
            subset_images.append(image_info)
            
            # Add all captions for this image
            subset_annotations.extend(captions_by_image[image_id])
            
            # Copy image if requested
            if copy_images:
                shutil.copy2(image_path, os.path.join(subset_image_dir, image_filename))
    
    # Create output structure
    output = {
        'info': {
            'description': 'COCO Caption Dataset Subset',
            'samples': len(subset_images),
            'total_captions': len(subset_annotations),
            'average_captions_per_image': len(subset_annotations) / len(subset_images)
        },
        'images': subset_images,
        'annotations': subset_annotations
    }
    
    # Save subset to JSON
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nCreated subset with:")
    print(f"- {len(subset_images)} images")
    print(f"- {len(subset_annotations)} total captions")
    print(f"- Average of {output['info']['average_captions_per_image']:.1f} captions per image")
    print(f"Saved to: {output_file}")
    
    if copy_images:
        print(f"Images copied to: {subset_image_dir}")
    
    # Print some example captions
    print("\nExample captions from the subset:")
    for i, image in enumerate(subset_images[:3]):  # Show first 3 images
        print(f"\nImage: {image['file_name']}")
        image_captions = [ann['caption'] for ann in subset_annotations if ann['image_id'] == image['id']]
        for j, caption in enumerate(image_captions, 1):
            print(f"Caption {j}: {caption}")
    
    return output

if __name__ == "__main__":
    # Set paths
    base_path = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/datasets/coco"  # Update this
    captions_file = os.path.join(base_path, "annotations/captions_val2014.json")
    images_dir = os.path.join(base_path, "val2014")
    n_samples = 200
    # Create subset
    subset_data = create_caption_subset(
        captions_file=captions_file,
        images_dir=images_dir,
        n_samples= n_samples,  # Number of images to sample
        output_file=f"caption_subset_{n_samples}_images.json",
        copy_images=True  # Set to True to copy images to a new directory
    )