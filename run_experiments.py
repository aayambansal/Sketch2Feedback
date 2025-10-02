"""
Main experiment runner for Sketch2Feedback project.
Runs all baselines and generates results.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import argparse
from tqdm import tqdm

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

def run_grammar_pipeline(dataset: List[Dict], dataset_type: str) -> List[Dict]:
    """Run grammar-in-the-loop pipeline on dataset."""
    print(f"Running grammar-in-the-loop pipeline on {dataset_type} dataset...")
    
    pipeline = GrammarInTheLoopPipeline()
    predictions = []
    
    for sample in tqdm(dataset):
        # Load image
        image_path = f"data/{dataset_type}_10/{sample['image_path']}"
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Process through pipeline
        result = pipeline.process_diagram(image_array, dataset_type)
        
        # Convert to prediction format
        prediction = {
            'error_types': [v.constraint_type for v in result['violations']],
            'bounding_boxes': [[p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]] 
                              for p in result['primitives']],
            'feedback': result['feedback']
        }
        
        predictions.append(prediction)
    
    return predictions

def run_end_to_end_lmm(dataset: List[Dict], dataset_type: str) -> List[Dict]:
    """Run end-to-end LMM baseline on dataset."""
    print(f"Running end-to-end LMM on {dataset_type} dataset...")
    
    model = EndToEndLMM()
    predictions = []
    
    for sample in tqdm(dataset):
        # Load image
        image_path = f"data/{dataset_type}_10/{sample['image_path']}"
        image = Image.open(image_path)
        
        # Make prediction
        pred = model.predict(image, dataset_type)
        
        # Convert to prediction format
        prediction = {
            'error_types': [pred.error_type] if pred.error_detected else [],
            'bounding_boxes': [list(pred.bbox)],
            'feedback': pred.feedback
        }
        
        predictions.append(prediction)
    
    return predictions

def run_vision_only_detector(dataset: List[Dict], dataset_type: str) -> List[Dict]:
    """Run vision-only detector baseline on dataset."""
    print(f"Running vision-only detector on {dataset_type} dataset...")
    
    model = VisionOnlyDetector()
    predictions = []
    
    for sample in tqdm(dataset):
        # Load image
        image_path = f"data/{dataset_type}_10/{sample['image_path']}"
        image = Image.open(image_path)
        
        # Make prediction
        pred = model.predict(image, dataset_type)
        
        # Convert to prediction format
        prediction = {
            'error_types': [pred.error_type] if pred.error_detected else [],
            'bounding_boxes': [list(pred.bbox)],
            'feedback': pred.feedback
        }
        
        predictions.append(prediction)
    
    return predictions

def evaluate_model(predictions: List[Dict], ground_truth: List[Dict], model_name: str) -> Dict[str, Any]:
    """Evaluate a model and return metrics."""
    print(f"Evaluating {model_name}...")
    
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.evaluate_model(predictions, ground_truth)
    
    # Generate report
    report = evaluator.generate_report(metrics, model_name)
    
    # Plot metrics
    plot_path = f"results/{model_name}_metrics.png"
    evaluator.plot_metrics(metrics, model_name, plot_path)
    
    return {
        'metrics': metrics,
        'report': report,
        'plot_path': plot_path
    }

def run_experiments():
    """Run all experiments."""
    print("Starting Sketch2Feedback experiments...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Generate datasets if they don't exist
    if not os.path.exists("data/fbd_10/metadata.json"):
        print("Generating FBD-10 dataset...")
        generate_fbd_dataset()
    
    if not os.path.exists("data/circuit_10/metadata.json"):
        print("Generating Circuit-10 dataset...")
        generate_circuit_dataset()
    
    # Load datasets
    fbd_dataset = load_dataset("data/fbd_10")
    circuit_dataset = load_dataset("data/circuit_10")
    
    print(f"Loaded {len(fbd_dataset)} FBD samples and {len(circuit_dataset)} Circuit samples")
    
    # Run experiments on FBD dataset
    print("\n" + "="*50)
    print("RUNNING EXPERIMENTS ON FBD-10 DATASET")
    print("="*50)
    
    # Grammar-in-the-loop
    fbd_grammar_preds = run_grammar_pipeline(fbd_dataset, "fbd")
    fbd_grammar_results = evaluate_model(fbd_grammar_preds, fbd_dataset, "Grammar-FBD")
    
    # End-to-end LMM
    fbd_lmm_preds = run_end_to_end_lmm(fbd_dataset, "fbd")
    fbd_lmm_results = evaluate_model(fbd_lmm_preds, fbd_dataset, "LMM-FBD")
    
    # Vision-only detector
    fbd_vision_preds = run_vision_only_detector(fbd_dataset, "fbd")
    fbd_vision_results = evaluate_model(fbd_vision_preds, fbd_dataset, "Vision-FBD")
    
    # Run experiments on Circuit dataset
    print("\n" + "="*50)
    print("RUNNING EXPERIMENTS ON CIRCUIT-10 DATASET")
    print("="*50)
    
    # Grammar-in-the-loop
    circuit_grammar_preds = run_grammar_pipeline(circuit_dataset, "circuit")
    circuit_grammar_results = evaluate_model(circuit_grammar_preds, circuit_dataset, "Grammar-Circuit")
    
    # End-to-end LMM
    circuit_lmm_preds = run_end_to_end_lmm(circuit_dataset, "circuit")
    circuit_lmm_results = evaluate_model(circuit_lmm_preds, circuit_dataset, "LMM-Circuit")
    
    # Vision-only detector
    circuit_vision_preds = run_vision_only_detector(circuit_dataset, "circuit")
    circuit_vision_results = evaluate_model(circuit_vision_preds, circuit_dataset, "Vision-Circuit")
    
    # Generate summary report
    generate_summary_report({
        'FBD': {
            'Grammar': fbd_grammar_results,
            'LMM': fbd_lmm_results,
            'Vision': fbd_vision_results
        },
        'Circuit': {
            'Grammar': circuit_grammar_results,
            'LMM': circuit_lmm_results,
            'Vision': circuit_vision_results
        }
    })
    
    print("\n" + "="*50)
    print("EXPERIMENTS COMPLETED")
    print("="*50)
    print("Results saved in 'results/' directory")

def generate_summary_report(results: Dict[str, Dict[str, Any]]):
    """Generate a summary report comparing all models."""
    print("Generating summary report...")
    
    # Create comparison table
    comparison_data = []
    
    for dataset_type, models in results.items():
        for model_name, result in models.items():
            metrics = result['metrics']
            comparison_data.append({
                'Dataset': dataset_type,
                'Model': model_name,
                'F1 Score': metrics.f1_score,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'Mean IoU': metrics.mean_iou,
                'Feedback Correctness': metrics.feedback_correctness,
                'Feedback Actionability': metrics.feedback_actionability,
                'Hallucination Rate': metrics.hallucination_rate,
                'Overall Score': metrics.overall_score
            })
    
    # Save comparison table
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    df.to_csv("results/comparison_table.csv", index=False)
    
    # Generate summary report
    report = "# Sketch2Feedback: Experimental Results Summary\n\n"
    
    report += "## Dataset Overview\n"
    report += "- **FBD-10**: 10 canonical physics scenarios with controllable errors\n"
    report += "- **Circuit-10**: 10 simple DC circuits with controllable errors\n\n"
    
    report += "## Model Comparison\n\n"
    report += "| Dataset | Model | F1 Score | Precision | Recall | Mean IoU | Feedback Correctness | Feedback Actionability | Hallucination Rate | Overall Score |\n"
    report += "|---------|-------|----------|-----------|--------|----------|---------------------|------------------------|-------------------|---------------|\n"
    
    for row in comparison_data:
        report += f"| {row['Dataset']} | {row['Model']} | {row['F1 Score']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['Mean IoU']:.3f} | {row['Feedback Correctness']:.3f} | {row['Feedback Actionability']:.3f} | {row['Hallucination Rate']:.3f} | {row['Overall Score']:.3f} |\n"
    
    report += "\n## Key Findings\n\n"
    
    # Find best performing models
    best_fbd = max([r for r in comparison_data if r['Dataset'] == 'FBD'], key=lambda x: x['Overall Score'])
    best_circuit = max([r for r in comparison_data if r['Dataset'] == 'Circuit'], key=lambda x: x['Overall Score'])
    
    report += f"### Best Performing Models\n"
    report += f"- **FBD Dataset**: {best_fbd['Model']} (Overall Score: {best_fbd['Overall Score']:.3f})\n"
    report += f"- **Circuit Dataset**: {best_circuit['Model']} (Overall Score: {best_circuit['Overall Score']:.3f})\n\n"
    
    # Grammar-in-the-loop analysis
    grammar_models = [r for r in comparison_data if 'Grammar' in r['Model']]
    if grammar_models:
        avg_grammar_score = np.mean([r['Overall Score'] for r in grammar_models])
        avg_grammar_hallucination = np.mean([r['Hallucination Rate'] for r in grammar_models])
        
        report += f"### Grammar-in-the-Loop Analysis\n"
        report += f"- **Average Overall Score**: {avg_grammar_score:.3f}\n"
        report += f"- **Average Hallucination Rate**: {avg_grammar_hallucination:.3f}\n\n"
    
    # End-to-end LMM analysis
    lmm_models = [r for r in comparison_data if 'LMM' in r['Model']]
    if lmm_models:
        avg_lmm_score = np.mean([r['Overall Score'] for r in lmm_models])
        avg_lmm_hallucination = np.mean([r['Hallucination Rate'] for r in lmm_models])
        
        report += f"### End-to-End LMM Analysis\n"
        report += f"- **Average Overall Score**: {avg_lmm_score:.3f}\n"
        report += f"- **Average Hallucination Rate**: {avg_lmm_hallucination:.3f}\n\n"
    
    # Vision-only analysis
    vision_models = [r for r in comparison_data if 'Vision' in r['Model']]
    if vision_models:
        avg_vision_score = np.mean([r['Overall Score'] for r in vision_models])
        avg_vision_hallucination = np.mean([r['Hallucination Rate'] for r in vision_models])
        
        report += f"### Vision-Only Detector Analysis\n"
        report += f"- **Average Overall Score**: {avg_vision_score:.3f}\n"
        report += f"- **Average Hallucination Rate**: {avg_vision_hallucination:.3f}\n\n"
    
    # Research question answers
    report += "## Research Question Answers\n\n"
    
    # RQ1: Error detection accuracy
    best_error_detection = max(comparison_data, key=lambda x: x['F1 Score'])
    report += f"### RQ1: Error Detection Accuracy\n"
    report += f"The {best_error_detection['Model']} model achieved the highest F1 score of {best_error_detection['F1 Score']:.3f} on the {best_error_detection['Dataset']} dataset.\n\n"
    
    # RQ2: Grammar-in-the-loop vs end-to-end
    if grammar_models and lmm_models:
        grammar_avg_f1 = np.mean([r['F1 Score'] for r in grammar_models])
        lmm_avg_f1 = np.mean([r['F1 Score'] for r in lmm_models])
        grammar_avg_hallucination = np.mean([r['Hallucination Rate'] for r in grammar_models])
        lmm_avg_hallucination = np.mean([r['Hallucination Rate'] for r in lmm_models])
        
        report += f"### RQ2: Grammar-in-the-Loop vs End-to-End LMM\n"
        report += f"- **Grammar-in-the-Loop F1**: {grammar_avg_f1:.3f}\n"
        report += f"- **End-to-End LMM F1**: {lmm_avg_f1:.3f}\n"
        report += f"- **Grammar-in-the-Loop Hallucination Rate**: {grammar_avg_hallucination:.3f}\n"
        report += f"- **End-to-End LMM Hallucination Rate**: {lmm_avg_hallucination:.3f}\n\n"
        
        if grammar_avg_hallucination < lmm_avg_hallucination:
            report += "**Result**: Grammar-in-the-loop approach reduces hallucinations compared to end-to-end LMM.\n\n"
        else:
            report += "**Result**: End-to-end LMM has lower hallucination rate than grammar-in-the-loop.\n\n"
    
    # RQ3: Neatness/quality effects
    report += f"### RQ3: Neatness/Quality Effects\n"
    report += "Analysis of neatness and scan quality effects requires additional experiments with varying image quality. This is a limitation of the current study.\n\n"
    
    # Save report
    with open("results/summary_report.md", 'w') as f:
        f.write(report)
    
    print("Summary report saved to results/summary_report.md")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Sketch2Feedback experiments")
    parser.add_argument("--dataset", choices=["fbd", "circuit", "both"], default="both",
                       help="Which dataset to run experiments on")
    parser.add_argument("--model", choices=["grammar", "lmm", "vision", "all"], default="all",
                       help="Which model to run")
    
    args = parser.parse_args()
    
    if args.dataset == "both" and args.model == "all":
        run_experiments()
    else:
        print("Custom experiment configurations not yet implemented.")
        print("Running full experiment suite...")
        run_experiments()

if __name__ == "__main__":
    main()
