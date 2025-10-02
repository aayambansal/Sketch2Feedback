"""
Evaluation script for Sketch2Feedback project.
"""

import os
import sys
import json

# Add src to path
sys.path.append('src')

from evaluation import ComprehensiveEvaluator

def load_results(results_dir: str):
    """Load results from JSON files."""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            model_name = filename.replace('.json', '')
            with open(os.path.join(results_dir, filename), 'r') as f:
                results[model_name] = json.load(f)
    
    return results

def main():
    """Main evaluation function."""
    print("Evaluating Sketch2Feedback results...")
    
    # Check if results exist
    if not os.path.exists("results"):
        print("No results found. Please run experiments first.")
        return
    
    # Load results
    results = load_results("results")
    
    if not results:
        print("No result files found in results directory.")
        return
    
    print(f"Found {len(results)} result files")
    
    # Generate evaluation report
    evaluator = ComprehensiveEvaluator()
    
    for model_name, result_data in results.items():
        print(f"Evaluating {model_name}...")
        
        # Extract predictions and ground truth
        predictions = result_data.get('predictions', [])
        ground_truth = result_data.get('ground_truth', [])
        
        if not predictions or not ground_truth:
            print(f"No data found for {model_name}")
            continue
        
        # Evaluate
        metrics = evaluator.evaluate_model(predictions, ground_truth)
        
        # Generate report
        report = evaluator.generate_report(metrics, model_name)
        
        # Save report
        report_path = f"results/{model_name}_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to {report_path}")
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
