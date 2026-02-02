import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
import json
import random

class BlipAdversarialGenerator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print("Loading BLIP model...")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # BLIP's normalization values
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def normalize_image(self, image_tensor):
        """Apply BLIP normalization"""
        return (image_tensor - self.mean) / self.std

    def denormalize_image(self, image_tensor):
        """Remove BLIP normalization"""
        return image_tensor * self.std + self.mean
    
    def get_image_features(self, image_tensor):
        """Extract BLIP vision features for an image tensor"""
        # Ensure the tensor requires grad
        image_tensor.requires_grad_(True)
        
        # Normalize the input
        normalized_tensor = self.normalize_image(image_tensor)
        
        # Get vision features
        vision_outputs = self.model.vision_model(pixel_values=normalized_tensor)
        image_features = vision_outputs.last_hidden_state
        # Pool features
        pooled_features = image_features.mean(dim=1)
        
        return pooled_features
    
    def optimize_image(self, source_tensor, target_emb, learning_rate=0.009, 
                      l2_dist_threshold=25, cosine_sim_threshold=0.95, max_iterations=10000):
        """Optimize source image to match target embedding using BLIP features"""
        # Initialize current input tensor
        cur_input = source_tensor.clone().to(self.device)
        cur_input.requires_grad_(True)
        
        # Keep track of best result
        best_loss = float('inf')
        best_input = cur_input.clone()
        
        # Initialize optimizer
        optimizer = torch.optim.Adam([cur_input], lr=learning_rate)
        
        # Keep original image for regularization
        source_copy = source_tensor.clone().detach()
        
        squared_l2_distance = float('inf')
        cosine_sim = 0
        iteration = 0
        
        while (squared_l2_distance >= l2_dist_threshold or 
               cosine_sim <= cosine_sim_threshold) and iteration < max_iterations:
            
            optimizer.zero_grad()
            
            # Get current features
            cur_emb = self.get_image_features(cur_input)
            
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
            
            # Clamp values to valid image range
            with torch.no_grad():
                cur_input.data.clamp_(0, 1)
            
            # Compute metrics
            with torch.no_grad():
                cur_emb = self.get_image_features(cur_input)
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
        
        return best_input.detach()
    
    def process_dataset(self, subset_json, images_dir, output_dir, n_targets=1):
        """Process the dataset and create adversarial samples using BLIP features"""
        # Load subset data
        with open(subset_json, 'r') as f:
            subset_data = json.load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare mapping dictionary
        adversarial_pairs = []
        
        # Create transform for loading images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Get list of all images
        all_images = subset_data['images']
        
        # Process each image
        for source_img_info in tqdm(all_images, desc="Generating adversarial samples"):
            source_filename = source_img_info['file_name']
            source_path = os.path.join(images_dir, source_filename)
            
            # Select random target images
            possible_targets = [img for img in all_images if img['id'] != source_img_info['id']]
            target_images = random.sample(possible_targets, min(n_targets, len(possible_targets)))
            
            try:
                # Load and preprocess source image
                source_img = Image.open(source_path).convert('RGB')
                source_tensor = transform(source_img).unsqueeze(0).to(self.device)
                
                # Process each target
                for target_img_info in target_images:
                    target_filename = target_img_info['file_name']
                    target_path = os.path.join(images_dir, target_filename)
                    
                    # Load and preprocess target image
                    target_img = Image.open(target_path).convert('RGB')
                    target_tensor = transform(target_img).unsqueeze(0).to(self.device)
                    
                    # Get target embedding
                    with torch.no_grad():
                        target_emb = self.get_image_features(target_tensor)
                    
                    # Optimize source image towards target
                    optimized_tensor = self.optimize_image(source_tensor, target_emb)
                    
                    # Create new filename with source@target format
                    source_base = os.path.splitext(source_filename)[0]
                    target_base = os.path.splitext(target_filename)[0]
                    adv_filename = f"{source_base}@{target_base}.png"
                    adv_path = os.path.join(output_dir, adv_filename)
                    
                    # Denormalize and save the image
                    # save_tensor = self.denormalize_image(optimized_tensor)
                    save_img = transforms.ToPILImage()(optimized_tensor.squeeze().cpu().clamp(0, 1))
                    save_img.save(adv_path)
                    
                    # Record the mapping
                    adversarial_pairs.append({
                        'source_image': source_filename,
                        'target_image': target_filename,
                        'adversarial_image': adv_filename
                    })
                    
            except Exception as e:
                print(f"Error processing {source_filename}: {str(e)}")
                continue
        
        # Save mapping to JSON
        mapping_file = os.path.join(output_dir, "adversarial_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump({
                'info': {
                    'total_pairs': len(adversarial_pairs),
                    'targets_per_image': n_targets
                },
                'pairs': adversarial_pairs
            }, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Generated {len(adversarial_pairs)} adversarial samples")
        print(f"Results saved to {output_dir}")
        print(f"Mapping file: {mapping_file}")

def main():
    # Set paths
    subset_json = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/caption_subset_200_images.json"
    images_dir = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/caption_subset_images"
    output_dir = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/caption_adversarial_output"
    
    # Initialize generator
    generator = BlipAdversarialGenerator()
    
    # Process dataset
    generator.process_dataset(
        subset_json=subset_json,
        images_dir=images_dir,
        output_dir=output_dir,
        n_targets=1
    )

if __name__ == "__main__":
    main()