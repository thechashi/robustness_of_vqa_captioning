import json
import random
from collections import defaultdict, Counter
from pathlib import Path
import os

def create_vqa_subset(questions_file, annotations_file, images_dir, n_samples=5, output_file="vqa_subset.json"):
    """
    Create a balanced subset of VQA samples with n examples for each of the top 10 question types.
    
    Args:
        questions_file: Path to VQA questions file
        annotations_file: Path to VQA annotations file
        images_dir: Directory containing the images
        n_samples: Number of samples per question type
        output_file: Output JSON file path
    """
    # Load data
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
        
    # Create question lookup
    question_lookup = {q['question_id']: q for q in questions['questions']}
    
    # Count question types
    question_types = Counter(ann['question_type'] for ann in annotations['annotations'])
    top_10_types = sorted(question_types.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 question types:")
    for qtype, count in top_10_types:
        print(f"{qtype}: {count} questions")
    
    # Group annotations by question type
    questions_by_type = defaultdict(list)
    for ann in annotations['annotations']:
        questions_by_type[ann['question_type']].append(ann)
    
    # Create subset
    subset = []
    for qtype, _ in top_10_types:
        # Get all questions of this type
        available_questions = questions_by_type[qtype]
        
        # Randomly sample n questions
        selected = random.sample(available_questions, min(n_samples, len(available_questions)))
        
        for ann in selected:
            question_info = question_lookup[ann['question_id']]
            image_id = question_info['image_id']
            image_filename = f'COCO_val2014_{image_id:012d}.jpg'
            image_path = os.path.join(images_dir, image_filename)
            
            # Only include if image exists
            if os.path.exists(image_path):
                subset_item = {
                    'question_id': ann['question_id'],
                    'image_id': image_id,
                    'image_filename': image_filename,
                    'question': question_info['question'],
                    'question_type': ann['question_type'],
                    'answers': ann['answers'],
                    'multiple_choice_answer': ann.get('multiple_choice_answer', '')
                }
                subset.append(subset_item)
    
    # Save subset to JSON
    output = {
        'info': {
            'description': 'VQA v2 validation subset',
            'samples_per_type': n_samples,
            'question_types': dict(top_10_types)
        },
        'samples': subset
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nCreated subset with {len(subset)} samples")
    print(f"Saved to: {output_file}")
    
    # Print distribution in subset
    subset_types = Counter(item['question_type'] for item in subset)
    print("\nDistribution in subset:")
    for qtype, count in subset_types.most_common():
        print(f"{qtype}: {count} samples")
    
    return output

# Example usage
if __name__ == "__main__":
    # Set paths
    base_path = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/datasets"
    questions_file = f"{base_path}/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
    annotations_file = f"{base_path}/vqa/v2_mscoco_val2014_annotations.json"
    images_dir = f"{base_path}/coco/val2014"
    n_samples = 20
    # Create subset with 5 samples per question type
    subset_data = create_vqa_subset(
        questions_file=questions_file,
        annotations_file=annotations_file,
        images_dir=images_dir,
        n_samples=20,
        output_file=f"vqa_subset_{n_samples}_per_type.json"
    )