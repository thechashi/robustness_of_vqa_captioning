import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
import json
import random

class BlipVQAAdversarialGenerator:
    def __init__(self, device="cuda:1" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print("Loading BLIP model...")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    def get_image_features(self, image):
        """Extract BLIP vision features for an image"""
        # Process image using BLIP's processor
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get vision features
        vision_outputs = self.model.vision_model(**inputs)
        image_features = vision_outputs.last_hidden_state
        # Pool features
        pooled_features = image_features.mean(dim=1)
        
        return pooled_features
        
    def optimize_image(self, source_img, target_emb, learning_rate=0.009, 
                l2_dist_threshold=36, cosine_sim_threshold=0.95, max_iterations=10000):
        """Optimize source image to match target embedding using BLIP features"""
        # Get initial BLIP processed tensor with gradient tracking
        inputs = self.processor(images=source_img, return_tensors="pt").to(self.device)
        cur_input = inputs.pixel_values
        cur_input.requires_grad_(True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam([cur_input], lr=learning_rate)
        
        # Keep original tensor for regularization
        source_copy = cur_input.clone().detach()
        
        squared_l2_distance = float('inf')
        cosine_sim = 0
        iteration = 0
        best_loss = float('inf')
        best_input = cur_input.clone()
        
        while (squared_l2_distance >= l2_dist_threshold or 
            cosine_sim <= cosine_sim_threshold) and iteration < max_iterations:
            
            optimizer.zero_grad()
            
            # Get vision features directly from tensor
            vision_outputs = self.model.vision_model(pixel_values=cur_input)
            cur_emb = vision_outputs.last_hidden_state.mean(dim=1)
            
            # Compute feature matching loss
            feature_loss = F.mse_loss(target_emb, cur_emb)
            
            # Add regularization to keep the image similar to source
            perturbation_loss = F.mse_loss(cur_input, source_copy)
            
            # Total loss with regularization
            total_loss = feature_loss + 0.1 * perturbation_loss
            
            # Backward pass
            total_loss.backward()
            
            # Update
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                vision_outputs = self.model.vision_model(pixel_values=cur_input)
                cur_emb = vision_outputs.last_hidden_state.mean(dim=1)
                squared_l2_distance = torch.sum((target_emb - cur_emb)**2).item()
                cosine_sim = F.cosine_similarity(target_emb, cur_emb).item()
            
            # Save best result
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_input = cur_input.clone()
            
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: L2 Distance = {squared_l2_distance:.6f}, "
                    f"Cosine Similarity = {cosine_sim:.6f}")
            
            iteration += 1
        
        return best_input
    def process_dataset(self, vqa_subset_json, images_dir, output_dir):
        """Process the VQA dataset and create adversarial samples using BLIP features"""
        # Load subset data
        with open(vqa_subset_json, 'r') as f:
            subset_data = json.load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of all samples
        all_samples = subset_data['samples']
        
        # Prepare mapping dictionary
        adversarial_pairs = []
        
        # Process each image
        for source_sample in tqdm(all_samples, desc="Generating adversarial samples"):
            source_filename = source_sample['image_filename']
            source_path = os.path.join(images_dir, source_filename)
            
            # Select random target image
            possible_targets = [sample for sample in all_samples 
                              if sample['image_id'] != source_sample['image_id']]
            target_sample = random.choice(possible_targets)
            target_filename = target_sample['image_filename']
            target_path = os.path.join(images_dir, target_filename)
            
            try:
                # Load and preprocess source image
                source_img = Image.open(source_path).convert('RGB')
                source_img = source_img.resize((224, 224))
                
                # Load and preprocess target image
                target_img = Image.open(target_path).convert('RGB')
                target_img = target_img.resize((224, 224))
                
                # Get target embedding using BLIP processor
                with torch.no_grad():
                    target_emb = self.get_image_features(target_img)
                
                # Optimize source image towards target
                optimized_tensor = self.optimize_image(source_img, target_emb)
                
                # Create new filename with source@target format
                source_base = os.path.splitext(source_filename)[0]
                target_base = os.path.splitext(target_filename)[0]
                adv_filename = f"{source_base}@{target_base}.png"
                adv_path = os.path.join(output_dir, adv_filename)
                
                # Save the image
                save_img = transforms.ToPILImage()(optimized_tensor.squeeze().cpu())
                save_img.save(adv_path)
                
                # Record the mapping with VQA-specific information
                adversarial_pairs.append({
                    'source_image': {
                        'filename': source_filename,
                        'question_id': source_sample['question_id'],
                        'question': source_sample['question'],
                        'question_type': source_sample['question_type'],
                        'answers': source_sample['answers']
                    },
                    'target_image': {
                        'filename': target_filename,
                        'question_id': target_sample['question_id'],
                        'question': target_sample['question'],
                        'question_type': target_sample['question_type'],
                        'answers': target_sample['answers']
                    },
                    'adversarial_image': adv_filename
                })
                
            except Exception as e:
                print(f"Error processing {source_filename}: {str(e)}")
                continue
        
        # Save mapping to JSON
        mapping_file = os.path.join(output_dir, "vqa_adversarial_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump({
                'info': {
                    'total_pairs': len(adversarial_pairs),
                    'description': 'VQA adversarial image pairs with questions and answers'
                },
                'pairs': adversarial_pairs
            }, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Generated {len(adversarial_pairs)} adversarial samples")
        print(f"Results saved to {output_dir}")
        print(f"Mapping file: {mapping_file}")

def main():
    # Set paths
    vqa_subset_json = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/vqa_subset_20_per_type.json"
    images_dir = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/datasets/coco/val2014"
    output_dir = "./vqa_adversarial_output"
    
    # Initialize generator
    generator = BlipVQAAdversarialGenerator()
    
    # Process dataset
    generator.process_dataset(
        vqa_subset_json=vqa_subset_json,
        images_dir=images_dir,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()