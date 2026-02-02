import os
import json
import random
import shutil
from PIL import Image
import torch
import numpy as np
from tqdm.notebook import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from bert_score import score as bert_score
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import clip
import nltk
nltk.download('punkt', download_dir='/home/chashi/nltk_data')
from nltk.tokenize import word_tokenize



class BLIP2Evaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Initialize BLIP-2
        print("Loading BLIP-2 model...")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = self.model.to(device)
        
        # Initialize CLIP for CLIP Score
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        
        # Initialize ROUGE
        self.rouge_scorer = ROUGEScore()
        
        # Initialize smoothing for BLEU
        self.smooth = SmoothingFunction()
    
    def prepare_subset(self, coco_root, num_images=10):
        """Prepare random subset of validation images"""
        # Load validation annotations
        ann_file = os.path.join(coco_root, "annotations/captions_val2014.json")
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Get unique image IDs
        image_info = {}
        for img in annotations['images']:
            image_info[img['id']] = img['file_name']
            
        img_to_captions = {}
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            if img_id in image_info:
                if image_info[img_id] not in img_to_captions:
                    img_to_captions[image_info[img_id]] = []
                img_to_captions[image_info[img_id]].append(ann['caption'])
        
        # Randomly select images
        selected_images = random.sample(list(img_to_captions.keys()), num_images)
        
        # Create subset directory
        subset_dir = os.path.join(coco_root, "eval_subset_blip2")
        os.makedirs(subset_dir, exist_ok=True)
        
        # Copy selected images
        selected_annotations = {}
        for img_file in selected_images:
            src_path = os.path.join(coco_root, "val2014", img_file)
            dst_path = os.path.join(subset_dir, img_file)
            
            # Copy image
            shutil.copy2(src_path, dst_path)
            
            # Store annotations
            selected_annotations[img_file] = img_to_captions[img_file]
        
        return subset_dir, selected_annotations
    
    def generate_caption(self, image_path):
        """Generate caption for single image"""
        image = Image.open(image_path)
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
    
    def compute_clip_score(self, image_path, caption):
        """Compute CLIP score between image and caption"""
        image = Image.open(image_path)
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([caption]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).item()
        
        return similarity
    
    def compute_bleu_score(self, hypotheses, references_list):
        """Compute BLEU score using NLTK"""
        references = [ref.split() for ref in references_list]
        hypothesis = hypotheses.split()
        
        weights = [(1.0,), (0.5, 0.5), (0.33, 0.33, 0.33), (0.25, 0.25, 0.25, 0.25)]
        bleu_scores = []
        
        for weight in weights:
            score = corpus_bleu([references], [hypothesis], 
                              weights=weight,
                              smoothing_function=self.smooth.method1)
            bleu_scores.append(score)
            
        return bleu_scores
        # return [0]*4
    
    def evaluate_subset(self, subset_dir, reference_annotations):
        """Evaluate the subset of images using multiple metrics"""
        results = []
        all_refs = []
        all_hyps = []
        
        for img_file in tqdm(os.listdir(subset_dir)):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(subset_dir, img_file)
            
            # try:
            # Generate caption
            generated_caption = self.generate_caption(img_path)
            references = reference_annotations[img_file]
            
            # Compute ROUGE scores
            rouge_scores = self.rouge_scorer(generated_caption, references[0])
            
            # Compute BERTScore
            P, R, F1 = bert_score([generated_caption], [references[0]], lang='en')
            
            # Compute CLIP Score
            clip_similarity = self.compute_clip_score(img_path, generated_caption)
            
            # Store for corpus BLEU computation
            all_refs.append(references)
            all_hyps.append(generated_caption)
            results.append({
                'image': img_file,
                'generated_caption': generated_caption,
                'references': references,
                'rouge_scores': {k: v.item() for k, v in rouge_scores.items()},
                'bert_score': {
                    'precision': P.item(),
                    'recall': R.item(),
                    'f1': F1.item()
                },
                'clip_score': clip_similarity
            })
            
            # except Exception as e:
            #     print(f"Error processing {img_file}: {str(e)}")
            #     continue
        
        # Compute corpus-level BLEU scores
        bleu_scores = []
        for i, hyp in enumerate(all_hyps):
            scores = self.compute_bleu_score(hyp, all_refs[i])
            bleu_scores.append(scores)
        
        avg_bleu_scores = np.mean(bleu_scores, axis=0)
        
        # Calculate overall metrics
        overall_metrics = {
            'bleu_scores': {f'bleu_{i+1}': score for i, score in enumerate(avg_bleu_scores)},
            'avg_clip_score': np.mean([r['clip_score'] for r in results]),
            'avg_bert_f1': np.mean([r['bert_score']['f1'] for r in results]),
            'avg_rouge_l_fmeasures': np.mean([r['rouge_scores']['rougeL_fmeasure'] for r in results]),
            'avg_rouge_l_precision': np.mean([r['rouge_scores']['rougeL_precision'] for r in results]),
            'avg_rouge_l_recall': np.mean([r['rouge_scores']['rougeL_recall'] for r in results])
        }
        
        return results, overall_metrics

def main():   
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Initialize evaluator
    evaluator = BLIP2Evaluator()
    
    # Prepare subset of images
    coco_root = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/datasets/coco"
    num_images = 10  # Adjust as needed
    subset_dir, reference_annotations = evaluator.prepare_subset(coco_root, num_images)
    
    # Run evaluation
    print(f"\nEvaluating {num_images} images...")
    results, overall_metrics = evaluator.evaluate_subset(subset_dir, reference_annotations)
    
    # Print results
    print("\nOverall Metrics:")
    for k, v in overall_metrics['bleu_scores'].items():
        print(f"{k.upper()}: {v:.4f}")
    print(f"Average CLIP Score: {overall_metrics['avg_clip_score']:.4f}")
    print(f"Average BERTScore F1: {overall_metrics['avg_bert_f1']:.4f}")
    print(f"Average ROUGE-L f-measures: {overall_metrics['avg_rouge_l_fmeasures']:.4f}")
    print(f"Average ROUGE-L precision: {overall_metrics['avg_rouge_l_precision']:.4f}")
    print(f"Average ROUGE-L recall: {overall_metrics['avg_rouge_l_recall']:.4f}")
    
    # Print individual results
    print("\nDetailed Results:")
    for result in results:
        print(f"\nImage: {result['image']}")
        print(f"Generated: {result['generated_caption']}")
        print(f"Reference: {result['references'][0]}")
        print(f"CLIP Score: {result['clip_score']:.4f}")
        print(f"BERTScore F1: {result['bert_score']['f1']:.4f}")
        print(f"ROUGE-L f-measure: {result['rouge_scores']['rougeL_fmeasure']:.4f}")
        print(f"ROUGE-L precision: {result['rouge_scores']['rougeL_precision']:.4f}")
        print(f"ROUGE-L recall: {result['rouge_scores']['rougeL_recall']:.4f}")

    # Save results to file
    output_file = "blip2_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'overall_metrics': overall_metrics,
            'individual_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()