import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
from collections import Counter, defaultdict
import re
import os

class SubsetVQAEvaluator:
    def __init__(self, device="cuda:1" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        print("Loading BLIP-2 model...")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = self.model.to(device)
    
    def preprocess_answer(self, answer):
        """Preprocess answer for consistent comparison"""
        if not isinstance(answer, str):
            print(f"Warning: Invalid answer type - {type(answer)}: {answer}")
            return ""
            
        answer = answer.lower()
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = ' '.join(answer.split())
        return answer
    
    def vqa_accuracy(self, pred, answers):
        """Calculate VQA accuracy"""
        if not answers:
            return 0.0
            
        pred = self.preprocess_answer(pred)
        processed_answers = [self.preprocess_answer(a['answer']) for a in answers]
        
        answer_count = Counter(processed_answers)
        if pred in answer_count:
            accuracy = min(answer_count[pred] / 3.0, 1.0)
            return accuracy
        return 0.0
    
    def generate_answer(self, image_path, question):
        """Generate answer for a single VQA pair"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            prompt = f"Answer the following question about the image with a brief response. Question: {question} Answer:"
            
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    min_length=1,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.5
                )
                
            answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            answer = answer.replace(prompt, '').strip()
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer for {image_path}: {str(e)}")
            return None

    def evaluate_subset(self, subset_file, image_dir):
        """Evaluate BLIP-2 on the subset of VQA data"""
        print(f"Loading subset from {subset_file}...")
        with open(subset_file, 'r') as f:
            subset_data = json.load(f)
        
        results = []
        question_type_metrics = defaultdict(list)
        processed = 0
        skipped = 0
        
        # Create two mappings for available images
        available_images = {}  # For files with @
        direct_images = set()  # For files without @
        
        for filename in os.listdir(image_dir):
            if '@' in filename:
                base_name = filename.split('@')[0]  # Get the part before @
                available_images[base_name] = filename
            else:
                direct_images.add(filename)
        
        # Process each sample
        samples = subset_data['samples']
        for sample in tqdm(samples, desc="Evaluating samples"):
            image_filename = sample['image_filename']
            original_filename = image_filename.split('.')[0]  # Remove extension
            
            # First check if the exact file exists
            if image_filename in direct_images:
                actual_filename = image_filename
            # Then check if there's a matching file with @
            elif original_filename in available_images:
                actual_filename = available_images[original_filename]
            else:
                print(f"Image not found: {image_filename}")
                skipped += 1
                continue
            
            image_path = os.path.join(image_dir, actual_filename)
            if not os.path.exists(image_path):
                print(f"Image path does not exist: {image_path}")
                skipped += 1
                continue
            
            # Generate answer
            generated_answer = self.generate_answer(image_path, sample['question'])
            if generated_answer is None:
                skipped += 1
                continue
            
            # Calculate accuracy
            accuracy = self.vqa_accuracy(generated_answer, sample['answers'])
            
            # Track metrics by question type
            question_type_metrics[sample['question_type']].append(accuracy)
            
            # Store result
            result = {
                'question_id': sample['question_id'],
                'image_id': sample['image_id'],
                'question': sample['question'],
                'question_type': sample['question_type'],
                'generated_answer': generated_answer,
                'ground_truth_answers': [a['answer'] for a in sample['answers']],
                'accuracy': accuracy,
                'image_filename': actual_filename
            }
            results.append(result)
            processed += 1
        
        # Calculate overall metrics
        all_accuracies = [r['accuracy'] for r in results]
        
        overall_metrics = {
            'mean_accuracy': float(np.mean(all_accuracies)),
            'median_accuracy': float(np.median(all_accuracies)),
            'total_processed': processed,
            'total_skipped': skipped,
            'question_type_metrics': {
                qtype: {
                    'mean_accuracy': float(np.mean(accs)),
                    'count': len(accs),
                    'accuracies': accs
                }
                for qtype, accs in question_type_metrics.items()
            }
        }
        
        return results, overall_metrics

def main():
    # Initialize evaluator
    evaluator = SubsetVQAEvaluator()
    
    # Set paths
    subset_file = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/vqa_subset_20_per_type.json"
    image_dir = "/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/vqa_subset_images"
    
    try:
        # Run evaluation
        results, metrics = evaluator.evaluate_subset(subset_file, image_dir)
        
        # Print results
        print("\nEvaluation Summary:")
        print(f"Total samples processed: {metrics['total_processed']}")
        print(f"Total samples skipped: {metrics['total_skipped']}")
        print(f"Overall Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"Overall Median Accuracy: {metrics['median_accuracy']:.4f}")
        
        print("\nAccuracy by Question Type:")
        for qtype, qmetrics in metrics['question_type_metrics'].items():
            print(f"\n{qtype}:")
            print(f"  Mean Accuracy: {qmetrics['mean_accuracy']:.4f}")
            print(f"  Number of samples: {qmetrics['count']}")
            
        # Save detailed results
        output_file = "subset_evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'overall_metrics': metrics,
                'individual_results': results
            }, f, indent=2)
            
        print(f"\nDetailed results saved to {output_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()