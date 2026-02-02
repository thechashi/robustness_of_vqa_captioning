import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import Counter
import re
import random
from pathlib import Path

class BLIP2VQAEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
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
        """Calculate VQA accuracy with detailed logging"""
        if not answers:
            print("Warning: Empty ground truth answers")
            return 0.0
            
        pred = self.preprocess_answer(pred)
        processed_answers = [self.preprocess_answer(a) for a in answers]
        
        print(f"\nAccuracy Calculation:")
        print(f"Predicted (raw): {pred}")
        print(f"Ground truth answers (raw): {answers}")
        print(f"Processed prediction: {pred}")
        print(f"Processed ground truth: {processed_answers}")
        
        answer_count = Counter(processed_answers)
        if pred in answer_count:
            accuracy = min(answer_count[pred] / 3.0, 1.0)
            print(f"Match found! Count: {answer_count[pred]}, Accuracy: {accuracy:.4f}")
            return accuracy
        else:
            print("No exact match found")
            return 0.0
    
    def generate_answer(self, image_path, question):
        """Generate answer for a single VQA pair with improved prompting"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Create a VQA-specific prompt that encourages direct answers
            prompt = (
                "Answer the following question about the image with a brief response. "
                "Question: " + question + " Answer:"
            )
            
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
                    max_length=200,  # Shorter max length to encourage concise answers
                    min_length=1,   # Ensure at least some output
                    num_beams=5,
                    length_penalty=1.0,  # Encourage shorter responses
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True,  # Enable sampling
                    top_p=0.9,       # Nucleus sampling
                    repetition_penalty=1.5  # Discourage repetition
                )
                
            answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            
            # Clean up the answer
            # Remove the question and prompt from the response if present
            answer = answer.replace(prompt, '').strip()
            # answer = answer.replace('Question:', '').strip()
            # answer = answer.replace('Answer:', '').strip()
            
            # # If the answer is too long or seems like a repetition, try to extract just the answer part
            # if len(answer.split()) > 10 or question.lower() in answer.lower():
            #     # Try to find the actual answer after "Answer:"
            #     if "Answer:" in answer:
            #         answer = answer.split("Answer:")[-1].strip()
            #     # Alternatively, take just the first sentence if it's too long
            #     else:
            #         answer = answer.split('.')[0].strip()
            
            print(f"\nGeneration Details:")
            print(f"Question: {question}")
            print(f"Generated Answer: {answer}")
            return answer
            
        except Exception as e:
            print(f"Error generating answer for {image_path}: {str(e)}")
            return None
    
    def evaluate_dataset(self, val_image_dir, questions_file, annotations_file, num_samples=None, seed=42):
        """Evaluate BLIP-2 on VQA v2 validation set"""
        # Resolve paths
        val_image_dir = os.path.expanduser(val_image_dir)
        questions_file = os.path.expanduser(questions_file)
        annotations_file = os.path.expanduser(annotations_file)
        
        # Verify files exist
        if not os.path.exists(val_image_dir):
            raise FileNotFoundError(f"Image directory not found: {val_image_dir}")
        if not os.path.exists(questions_file):
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
            
        print(f"\nDataset Paths:")
        print(f"Image directory: {val_image_dir}")
        print(f"Questions file: {questions_file}")
        print(f"Annotations file: {annotations_file}")
        
        # Load data
        print("\nLoading questions and annotations...")
        with open(questions_file, 'r') as f:
            questions = json.load(f)
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
            
        print(f"Total available questions: {len(questions['questions'])}")
        
        # Set random seed
        random.seed(seed)
        
        # Create lookups
        question_lookup = {q['question_id']: q for q in questions['questions']}
        
        # Select samples
        all_annotations = annotations['annotations']
        if num_samples is not None:
            all_annotations = random.sample(all_annotations, min(num_samples, len(all_annotations)))
            print(f"Selected {len(all_annotations)} samples for evaluation")
        
        results = []
        accuracies = []
        question_type_metrics = {}
        processed_samples = 0
        skipped_samples = 0
        
        # Process samples
        pbar = tqdm(all_annotations, desc="Processing samples")
        for ann in pbar:
            print(f"\n{'='*80}\nProcessing new sample:")
            
            question_id = ann['question_id']
            if question_id not in question_lookup:
                print(f"Question ID {question_id} not found in questions file")
                skipped_samples += 1
                continue
                
            question_info = question_lookup[question_id]
            image_id = question_info['image_id']
            question = question_info['question']
            
            # Get question type from annotation if available
            question_type = ann.get('question_type', 'unknown')
            
            print(f"Question ID: {question_id}")
            print(f"Question Type: {question_type}")
            print(f"Question: {question}")
            
            # Find image
            image_filename = f'COCO_val2014_{image_id:012d}.jpg'
            image_path = os.path.join(val_image_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                skipped_samples += 1
                continue
            
            # Generate answer
            generated_answer = self.generate_answer(image_path, question)
            if generated_answer is None:
                skipped_samples += 1
                continue
            
            # Get ground truth answers
            ground_truth = [a['answer'] for a in ann['answers']]
            print(f"Ground Truth Answers: {ground_truth}")
            
            # Calculate accuracy
            accuracy = self.vqa_accuracy(generated_answer, ground_truth)
            accuracies.append(accuracy)
            
            # Track metrics
            if question_type not in question_type_metrics:
                question_type_metrics[question_type] = []
            question_type_metrics[question_type].append(accuracy)
            
            # Store result
            results.append({
                'question_id': question_id,
                'image_id': image_id,
                'question': question,
                'question_type': question_type,
                'generated_answer': generated_answer,
                'ground_truth_answers': ground_truth,
                'accuracy': accuracy
            })
            
            processed_samples += 1
            print(f"Current sample accuracy: {accuracy:.4f}")
            pbar.set_postfix({'processed': processed_samples, 'skipped': skipped_samples, 'last_acc': accuracy})
        
        if not accuracies:
            print("No samples were successfully processed!")
            return None, None
        
        # Calculate metrics
        overall_metrics = {
            'mean_accuracy': float(np.mean(accuracies)),
            'median_accuracy': float(np.median(accuracies)),
            'total_processed_samples': processed_samples,
            'total_skipped_samples': skipped_samples,
            'question_type_metrics': {
                qtype: {
                    'mean_accuracy': float(np.mean(accs)),
                    'count': len(accs)
                }
                for qtype, accs in question_type_metrics.items()
            }
        }
        
        return results, overall_metrics

def main():
    # Initialize evaluator
    evaluator = BLIP2VQAEvaluator()
    
    # Set paths
    base_path = os.path.expanduser("/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/datasets/vqa")
    val_image_dir = os.path.expanduser("/home/chashi/Desktop/Research/My Projects/robustness_of_vqa_and_captioning_models/datasets/coco/val2014")
    questions_file = os.path.join(base_path, "v2_OpenEnded_mscoco_val2014_questions.json")
    annotations_file = os.path.join(base_path, "v2_mscoco_val2014_annotations.json")
    
    # Set number of samples
    num_samples = 5  # Small number for testing
    
    try:
        print(f"Evaluating BLIP-2 on VQA v2 validation set {'(subset)' if num_samples else '(full)'}")
        results, overall_metrics = evaluator.evaluate_dataset(
            val_image_dir,
            questions_file,
            annotations_file,
            num_samples=num_samples
        )
        
        if results is None or overall_metrics is None:
            print("Evaluation failed - no results generated")
            return
        
        print("\nEvaluation Summary:")
        print(f"Total samples processed: {overall_metrics['total_processed_samples']}")
        print(f"Total samples skipped: {overall_metrics['total_skipped_samples']}")
        print(f"Mean Accuracy: {overall_metrics['mean_accuracy']:.4f}")
        print(f"Median Accuracy: {overall_metrics['median_accuracy']:.4f}")
        
        print("\nQuestion Type Metrics:")
        for qtype, metrics in overall_metrics['question_type_metrics'].items():
            print(f"{qtype}:")
            print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
            print(f"  Count: {metrics['count']}")
        
        # Save results
        output_file = "blip2_vqa_evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'overall_metrics': overall_metrics,
                'individual_results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()