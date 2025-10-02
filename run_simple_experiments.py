"""
Simplified experiment runner for Sketch2Feedback project.
Runs a subset of experiments to generate results quickly.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import random

# Add src to path
sys.path.append('src')

from dataset_generator import generate_fbd_dataset, generate_circuit_dataset
from grammar_pipeline import GrammarInTheLoopPipeline
from baselines import EndToEndLMM, VisionOnlyDetector, BaselineEvaluator
from evaluation import ComprehensiveEvaluator

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(f"{dataset_path}/metadata.json", 'r') as f:
        return json.load(f)

def run_simple_experiments():
    """Run simplified experiments."""
    print("Starting simplified Sketch2Feedback experiments...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load a small subset of the dataset for quick testing
    fbd_dataset = load_dataset("data/fbd_10")
    circuit_dataset = load_dataset("data/circuit_10")
    
    # Use only first 5 samples from each dataset for quick testing
    fbd_subset = fbd_dataset[:5]
    circuit_subset = circuit_dataset[:5]
    
    print(f"Using {len(fbd_subset)} FBD samples and {len(circuit_subset)} Circuit samples")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Test Grammar-in-the-loop on FBD
    print("\nTesting Grammar-in-the-loop on FBD...")
    pipeline = GrammarInTheLoopPipeline()
    fbd_grammar_preds = []
    
    for sample in fbd_subset:
        try:
            # Load image
            image_path = f"data/fbd_10/{sample['image_path']}"
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Process through pipeline
            result = pipeline.process_diagram(image_array, "fbd")
            
            # Convert to prediction format
            prediction = {
                'error_types': [v.constraint_type for v in result['violations']],
                'bounding_boxes': [[p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]] 
                                  for p in result['primitives']],
                'feedback': result['feedback']
            }
            
            fbd_grammar_preds.append(prediction)
        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
            # Add dummy prediction
            fbd_grammar_preds.append({
                'error_types': [],
                'bounding_boxes': [],
                'feedback': "Error processing image"
            })
    
    # Test Vision-only detector on FBD
    print("Testing Vision-only detector on FBD...")
    vision_model = VisionOnlyDetector()
    fbd_vision_preds = []
    
    for sample in fbd_subset:
        try:
            # Load image
            image_path = f"data/fbd_10/{sample['image_path']}"
            image = Image.open(image_path)
            
            # Make prediction
            pred = vision_model.predict(image, "fbd")
            
            # Convert to prediction format
            prediction = {
                'error_types': [pred.error_type] if pred.error_detected else [],
                'bounding_boxes': [list(pred.bbox)],
                'feedback': pred.feedback
            }
            
            fbd_vision_preds.append(prediction)
        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
            # Add dummy prediction
            fbd_vision_preds.append({
                'error_types': [],
                'bounding_boxes': [],
                'feedback': "Error processing image"
            })
    
    # Test Grammar-in-the-loop on Circuit
    print("Testing Grammar-in-the-loop on Circuit...")
    circuit_grammar_preds = []
    
    for sample in circuit_subset:
        try:
            # Load image
            image_path = f"data/circuit_10/{sample['image_path']}"
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Process through pipeline
            result = pipeline.process_diagram(image_array, "circuit")
            
            # Convert to prediction format
            prediction = {
                'error_types': [v.constraint_type for v in result['violations']],
                'bounding_boxes': [[p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]] 
                                  for p in result['primitives']],
                'feedback': result['feedback']
            }
            
            circuit_grammar_preds.append(prediction)
        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
            # Add dummy prediction
            circuit_grammar_preds.append({
                'error_types': [],
                'bounding_boxes': [],
                'feedback': "Error processing image"
            })
    
    # Test Vision-only detector on Circuit
    print("Testing Vision-only detector on Circuit...")
    circuit_vision_preds = []
    
    for sample in circuit_subset:
        try:
            # Load image
            image_path = f"data/circuit_10/{sample['image_path']}"
            image = Image.open(image_path)
            
            # Make prediction
            pred = vision_model.predict(image, "circuit")
            
            # Convert to prediction format
            prediction = {
                'error_types': [pred.error_type] if pred.error_detected else [],
                'bounding_boxes': [list(pred.bbox)],
                'feedback': pred.feedback
            }
            
            circuit_vision_preds.append(prediction)
        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
            # Add dummy prediction
            circuit_vision_preds.append({
                'error_types': [],
                'bounding_boxes': [],
                'feedback': "Error processing image"
            })
    
    # Evaluate models
    print("\nEvaluating models...")
    
    # Evaluate FBD models
    fbd_grammar_metrics = evaluator.evaluate_model(fbd_grammar_preds, fbd_subset)
    fbd_vision_metrics = evaluator.evaluate_model(fbd_vision_preds, fbd_subset)
    
    # Evaluate Circuit models
    circuit_grammar_metrics = evaluator.evaluate_model(circuit_grammar_preds, circuit_subset)
    circuit_vision_metrics = evaluator.evaluate_model(circuit_vision_preds, circuit_subset)
    
    # Generate results
    results = {
        'FBD': {
            'Grammar': {
                'metrics': fbd_grammar_metrics,
                'predictions': fbd_grammar_preds,
                'ground_truth': fbd_subset
            },
            'Vision': {
                'metrics': fbd_vision_metrics,
                'predictions': fbd_vision_preds,
                'ground_truth': fbd_subset
            }
        },
        'Circuit': {
            'Grammar': {
                'metrics': circuit_grammar_metrics,
                'predictions': circuit_grammar_preds,
                'ground_truth': circuit_subset
            },
            'Vision': {
                'metrics': circuit_vision_metrics,
                'predictions': circuit_vision_preds,
                'ground_truth': circuit_subset
            }
        }
    }
    
    # Save results
    with open("results/simple_experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary report
    generate_simple_report(results)
    
    print("\nSimplified experiments completed!")
    print("Results saved in 'results/' directory")

def generate_simple_report(results: Dict[str, Dict[str, Any]]):
    """Generate a simple summary report."""
    print("Generating summary report...")
    
    report = "# Sketch2Feedback: Simplified Experimental Results\n\n"
    
    report += "## Dataset Overview\n"
    report += "- **FBD-10**: 5 samples from 10 canonical physics scenarios\n"
    report += "- **Circuit-10**: 5 samples from 10 simple DC circuits\n\n"
    
    report += "## Results Summary\n\n"
    
    for dataset_type, models in results.items():
        report += f"### {dataset_type} Dataset\n\n"
        
        for model_name, result in models.items():
            metrics = result['metrics']
            report += f"#### {model_name} Model\n"
            report += f"- **F1 Score**: {metrics.f1_score:.3f}\n"
            report += f"- **Precision**: {metrics.precision:.3f}\n"
            report += f"- **Recall**: {metrics.recall:.3f}\n"
            report += f"- **Mean IoU**: {metrics.mean_iou:.3f}\n"
            report += f"- **Feedback Correctness**: {metrics.feedback_correctness:.3f}\n"
            report += f"- **Feedback Actionability**: {metrics.feedback_actionability:.3f}\n"
            report += f"- **Hallucination Rate**: {metrics.hallucination_rate:.3f}\n"
            report += f"- **Overall Score**: {metrics.overall_score:.3f}\n\n"
    
    # Key findings
    report += "## Key Findings\n\n"
    
    # Find best performing models
    all_results = []
    for dataset_type, models in results.items():
        for model_name, result in models.items():
            all_results.append({
                'dataset': dataset_type,
                'model': model_name,
                'overall_score': result['metrics'].overall_score,
                'f1_score': result['metrics'].f1_score,
                'hallucination_rate': result['metrics'].hallucination_rate
            })
    
    best_overall = max(all_results, key=lambda x: x['overall_score'])
    best_f1 = max(all_results, key=lambda x: x['f1_score'])
    lowest_hallucination = min(all_results, key=lambda x: x['hallucination_rate'])
    
    report += f"### Best Overall Performance\n"
    report += f"- **Model**: {best_overall['model']} on {best_overall['dataset']} dataset\n"
    report += f"- **Overall Score**: {best_overall['overall_score']:.3f}\n\n"
    
    report += f"### Best Error Detection\n"
    report += f"- **Model**: {best_f1['model']} on {best_f1['dataset']} dataset\n"
    report += f"- **F1 Score**: {best_f1['f1_score']:.3f}\n\n"
    
    report += f"### Lowest Hallucination Rate\n"
    report += f"- **Model**: {lowest_hallucination['model']} on {lowest_hallucination['dataset']} dataset\n"
    report += f"- **Hallucination Rate**: {lowest_hallucination['hallucination_rate']:.3f}\n\n"
    
    # Research question answers
    report += "## Research Question Answers\n\n"
    
    report += f"### RQ1: Error Detection Accuracy\n"
    report += f"The {best_f1['model']} model achieved the highest F1 score of {best_f1['f1_score']:.3f} on the {best_f1['dataset']} dataset.\n\n"
    
    report += f"### RQ2: Grammar-in-the-Loop vs Vision-Only\n"
    grammar_results = [r for r in all_results if 'Grammar' in r['model']]
    vision_results = [r for r in all_results if 'Vision' in r['model']]
    
    if grammar_results and vision_results:
        avg_grammar_f1 = np.mean([r['f1_score'] for r in grammar_results])
        avg_vision_f1 = np.mean([r['f1_score'] for r in vision_results])
        avg_grammar_hallucination = np.mean([r['hallucination_rate'] for r in grammar_results])
        avg_vision_hallucination = np.mean([r['hallucination_rate'] for r in vision_results])
        
        report += f"- **Grammar-in-the-Loop F1**: {avg_grammar_f1:.3f}\n"
        report += f"- **Vision-Only F1**: {avg_vision_f1:.3f}\n"
        report += f"- **Grammar-in-the-Loop Hallucination Rate**: {avg_grammar_hallucination:.3f}\n"
        report += f"- **Vision-Only Hallucination Rate**: {avg_vision_hallucination:.3f}\n\n"
        
        if avg_grammar_hallucination < avg_vision_hallucination:
            report += "**Result**: Grammar-in-the-loop approach shows lower hallucination rate.\n\n"
        else:
            report += "**Result**: Vision-only approach shows lower hallucination rate.\n\n"
    
    report += f"### RQ3: Neatness/Quality Effects\n"
    report += "Analysis of neatness and scan quality effects requires additional experiments with varying image quality. This is a limitation of the current study.\n\n"
    
    # Save report
    with open("results/simple_summary_report.md", 'w') as f:
        f.write(report)
    
    print("Summary report saved to results/simple_summary_report.md")

if __name__ == "__main__":
    run_simple_experiments()
