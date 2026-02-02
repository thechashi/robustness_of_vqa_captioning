import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from bert_score import score as bert_score
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import clip
import nltk
import json
import os
from tqdm import tqdm
import numpy as np

class CaptionEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Initialize BLIP-2
        print("Loading BLIP-2 model...")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = self.model.to(device)
        
        # Initialize CLIP
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        
        # Initialize ROUGE and BLEU
        self.rouge_scorer = ROUGEScore()
        self.smooth = SmoothingFunction()
        
        nltk.download('punkt', quiet=True)
    
    def generate_caption(self, image_path):
        """Generate caption for a single image"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption
    
    def compute_metrics(self, generated_caption, reference_captions, image_path):
        """Compute all evaluation metrics for a single image"""
        # ROUGE scores
        rouge_scores = self.rouge_scorer(generated_caption, reference_captions[0])
        
        # BERTScore
        P, R, F1 = bert_score([generated_caption], [reference_captions[0]], lang='en')
        
        # CLIP Score
        image = Image.open(image_path).convert('RGB')
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([generated_caption]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            clip_similarity = (100.0 * image_features @ text_features.T).item()
        
        # BLEU scores
        references = [ref.split() for ref in reference_captions]
        hypothesis = generated_caption.split()
        
        weights = [(1.0,), (0.5, 0.5), (0.33, 0.33, 0.33), (0.25, 0.25, 0.25, 0.25)]
        bleu_scores = []
        
        for weight in weights:
            score = corpus_bleu([references], [hypothesis], 
                              weights=weight,
                              smoothing_function=self.smooth.method1)
            bleu_scores.append(score)
        
        return {
            'rouge_scores': {k: v.item() for k, v in rouge_scores.items()},
            'bert_score': {
                'precision': P.item(),
                'recall': R.item(),
                'f1': F1.item()
            },
            'clip_score': clip_similarity,
            'bleu_scores': {f'bleu_{i+1}': score for i, score in enumerate(bleu_scores)}
        }

    def compute_image_similarity(self, image_path1, image_path2):
        """Compute cosine similarity between CLIP embeddings of two images"""
        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')
        
        image1 = self.clip_preprocess(image1).unsqueeze(0).to(self.device)
        image2 = self.clip_preprocess(image2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image1_features = self.clip_model.encode_image(image1)
            image2_features = self.clip_model.encode_image(image2)
            
            image1_features /= image1_features.norm(dim=-1, keepdim=True)
            image2_features /= image2_features.norm(dim=-1, keepdim=True)
            
            similarity = (image1_features @ image2_features.T).item()
        
        return similarity
    def evaluate_dataset(self, image_dir, adversarial_dir, annotations_file):
        """Evaluate both original and adversarial images"""
        # Load annotations
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # Create lookup for image captions
        captions_lookup = {}
        for ann in data['annotations']:
            img_info = next(img for img in data['images'] if img['id'] == ann['image_id'])
            img_name = img_info['file_name']
            if img_name not in captions_lookup:
                captions_lookup[img_name] = []
            captions_lookup[img_name].append(ann['caption'])
        
        results = []
        # Get list of all original images
        original_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))]
        
        for orig_file in tqdm(original_files, desc="Processing image pairs"):
            try:
                orig_path = os.path.join(image_dir, orig_file)
                orig_base = orig_file.split('.')[0]
                
                # Find corresponding adversarial images
                adv_files = [f for f in os.listdir(adversarial_dir) if f.startswith(orig_base + '@')]
                
                for adv_file in adv_files:
                    adv_path = os.path.join(adversarial_dir, adv_file)
                    target_name = adv_file.split('@')[1].split('.')[0] + '.jpg'
                    target_path = os.path.join(image_dir, target_name)
                    
                    # Generate captions
                    orig_caption = self.generate_caption(orig_path)
                    adv_caption = self.generate_caption(adv_path)
                    target_caption = self.generate_caption(target_path)
                    
                    # Compute similarities
                    orig_adv_similarity = self.compute_image_similarity(orig_path, adv_path)
                    adv_target_similarity = self.compute_image_similarity(adv_path, target_path)
                    
                    # Compute metrics for original image
                    orig_metrics = self.compute_metrics(
                        orig_caption,
                        captions_lookup[orig_file],
                        orig_path
                    )
                    
                    # Compute metrics for adversarial image
                    adv_metrics = self.compute_metrics(
                        adv_caption,
                        captions_lookup[orig_file],  # Using original image's captions
                        adv_path
                    )
                    
                    results.append({
                        'image': orig_file,
                        'adversarial_image': adv_file,
                        'target_image': target_name,
                        'generated_original_caption': orig_caption,
                        'generated_adversarial_caption': adv_caption,
                        'generated_target_caption': target_caption,
                        'references': captions_lookup[orig_file],
                        'original_metrics': orig_metrics,
                        'adversarial_metrics': adv_metrics,
                        'similarities': {
                            'original_adversarial_similarity': orig_adv_similarity,
                            'adversarial_target_similarity': adv_target_similarity
                        }
                    })
                    
            except Exception as e:
                print(f"Error processing {orig_file}: {str(e)}")
                continue
        
        return results

def main():
    # Set paths
    base_dir = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models"
    original_dir = os.path.join(base_dir, "caption_subset_images")
    adversarial_dir = os.path.join(base_dir, "caption_adversarial_output")
    annotations_file = os.path.join(base_dir, "caption_subset_200_images.json")
    
    # Initialize evaluator
    evaluator = CaptionEvaluator()
    
    # Evaluate images
    print("\nEvaluating image pairs...")
    results = evaluator.evaluate_dataset(original_dir, adversarial_dir, annotations_file)
    
    # Calculate overall metrics
    def calculate_averages(results):
        orig_metrics = []
        adv_metrics = []
        similarities = []
        
        for r in results:
            orig_metrics.append(r['original_metrics'])
            adv_metrics.append(r['adversarial_metrics'])
            similarities.append(r['similarities'])
        
        return {
            'original_images': {
                'avg_clip_score': np.mean([m['clip_score'] for m in orig_metrics]),
                'avg_bert_f1': np.mean([m['bert_score']['f1'] for m in orig_metrics]),
                'avg_rouge_l': np.mean([m['rouge_scores']['rougeL_fmeasure'] for m in orig_metrics]),
                'avg_bleu_1': np.mean([m['bleu_scores']['bleu_1'] for m in orig_metrics]),
                'avg_bleu_4': np.mean([m['bleu_scores']['bleu_4'] for m in orig_metrics])
            },
            'adversarial_images': {
                'avg_clip_score': np.mean([m['clip_score'] for m in adv_metrics]),
                'avg_bert_f1': np.mean([m['bert_score']['f1'] for m in adv_metrics]),
                'avg_rouge_l': np.mean([m['rouge_scores']['rougeL_fmeasure'] for m in adv_metrics]),
                'avg_bleu_1': np.mean([m['bleu_scores']['bleu_1'] for m in adv_metrics]),
                'avg_bleu_4': np.mean([m['bleu_scores']['bleu_4'] for m in adv_metrics])
            },
            'similarities': {
                'avg_original_adversarial_similarity': np.mean([s['original_adversarial_similarity'] for s in similarities]),
                'avg_adversarial_target_similarity': np.mean([s['adversarial_target_similarity'] for s in similarities])
            }
        }
    
    overall_metrics = calculate_averages(results)
    
    # Save results
    output = {
        'overall_metrics': overall_metrics,
        'individual_results': results
    }
    
    output_file = "caption_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\nOverall Metrics:")
    print("\nOriginal Images:")
    for metric, value in overall_metrics['original_images'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAdversarial Images:")
    for metric, value in overall_metrics['adversarial_images'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nSimilarities:")
    for metric, value in overall_metrics['similarities'].items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()