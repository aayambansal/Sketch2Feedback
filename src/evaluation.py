"""
Evaluation suite for Sketch2Feedback project.
Measures error detection, localization, feedback quality, and hallucinations.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from scipy.spatial.distance import cdist
import pandas as pd

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Error detection metrics
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    
    # Localization metrics
    iou_at_05: float
    iou_at_075: float
    mean_iou: float
    
    # Feedback quality metrics
    feedback_correctness: float
    feedback_actionability: float
    feedback_relevance: float
    
    # Hallucination metrics
    hallucination_rate: float
    false_positive_rate: float
    
    # Overall metrics
    overall_score: float

class ErrorDetector:
    """Detects and evaluates errors in predictions vs ground truth."""
    
    def __init__(self):
        self.error_types = [
            "missing_force", "wrong_direction", "extra_force", "wrong_label",
            "missing_ground", "wrong_polarity", "wrong_connection", "missing_component"
        ]
    
    def detect_errors(self, prediction: Dict, ground_truth: Dict) -> Dict[str, Any]:
        """Detect errors in prediction compared to ground truth."""
        pred_errors = set(prediction.get('error_types', []))
        gt_errors = set(ground_truth.get('error_types', []))
        
        # Calculate error detection metrics
        true_positives = len(pred_errors.intersection(gt_errors))
        false_positives = len(pred_errors - gt_errors)
        false_negatives = len(gt_errors - pred_errors)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predicted_errors': list(pred_errors),
            'ground_truth_errors': list(gt_errors)
        }

class LocalizationEvaluator:
    """Evaluates localization accuracy of detected errors."""
    
    def __init__(self):
        self.iou_thresholds = [0.5, 0.75]
    
    def evaluate_localization(self, prediction: Dict, ground_truth: Dict) -> Dict[str, float]:
        """Evaluate localization accuracy using IoU."""
        pred_bboxes = prediction.get('bounding_boxes', [])
        gt_bboxes = ground_truth.get('bounding_boxes', [])
        
        if not pred_bboxes or not gt_bboxes:
            return {
                'iou_at_05': 0.0,
                'iou_at_075': 0.0,
                'mean_iou': 0.0
            }
        
        # Calculate IoU for each prediction
        ious = []
        for pred_bbox in pred_bboxes:
            max_iou = 0
            for gt_bbox in gt_bboxes:
                iou = self._calculate_iou(pred_bbox, gt_bbox)
                max_iou = max(max_iou, iou)
            ious.append(max_iou)
        
        # Calculate metrics
        iou_at_05 = sum(1 for iou in ious if iou >= 0.5) / len(ious) if ious else 0
        iou_at_075 = sum(1 for iou in ious if iou >= 0.75) / len(ious) if ious else 0
        mean_iou = np.mean(ious) if ious else 0
        
        return {
            'iou_at_05': iou_at_05,
            'iou_at_075': iou_at_075,
            'mean_iou': mean_iou
        }
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class FeedbackEvaluator:
    """Evaluates feedback quality using rubric-based metrics."""
    
    def __init__(self):
        self.rubric_criteria = {
            'correctness': {
                'description': 'Does the feedback correctly identify the actual error?',
                'scale': [1, 2, 3, 4, 5]
            },
            'actionability': {
                'description': 'Does the feedback suggest a fix that a novice can follow?',
                'scale': [1, 2, 3, 4, 5]
            },
            'relevance': {
                'description': 'Is the feedback relevant to the specific error?',
                'scale': [1, 2, 3, 4, 5]
            }
        }
    
    def evaluate_feedback(self, prediction: Dict, ground_truth: Dict) -> Dict[str, float]:
        """Evaluate feedback quality using rubric."""
        feedback = prediction.get('feedback', '')
        gt_errors = ground_truth.get('error_types', [])
        
        # Correctness: Does feedback identify the actual error?
        correctness = self._evaluate_correctness(feedback, gt_errors)
        
        # Actionability: Does feedback suggest actionable steps?
        actionability = self._evaluate_actionability(feedback)
        
        # Relevance: Is feedback relevant to the error?
        relevance = self._evaluate_relevance(feedback, gt_errors)
        
        return {
            'correctness': correctness,
            'actionability': actionability,
            'relevance': relevance
        }
    
    def _evaluate_correctness(self, feedback: str, gt_errors: List[str]) -> float:
        """Evaluate if feedback correctly identifies the error."""
        if not gt_errors:
            # No errors in ground truth
            if "correct" in feedback.lower() or "no error" in feedback.lower():
                return 5.0
            else:
                return 1.0
        
        # Check if feedback mentions the actual errors
        feedback_lower = feedback.lower()
        correct_mentions = 0
        
        for error in gt_errors:
            if self._feedback_mentions_error(feedback_lower, error):
                correct_mentions += 1
        
        # Score based on how many errors are correctly identified
        if correct_mentions == len(gt_errors):
            return 5.0
        elif correct_mentions > 0:
            return 3.0 + (correct_mentions / len(gt_errors)) * 2.0
        else:
            return 1.0
    
    def _evaluate_actionability(self, feedback: str) -> float:
        """Evaluate if feedback is actionable."""
        feedback_lower = feedback.lower()
        
        # Check for actionable keywords
        actionable_keywords = [
            "add", "remove", "check", "make sure", "ensure", "draw", "label",
            "connect", "orient", "direction", "polarity", "ground"
        ]
        
        actionability_score = 0
        for keyword in actionable_keywords:
            if keyword in feedback_lower:
                actionability_score += 1
        
        # Normalize to 1-5 scale
        if actionability_score >= 3:
            return 5.0
        elif actionability_score >= 2:
            return 4.0
        elif actionability_score >= 1:
            return 3.0
        else:
            return 2.0
    
    def _evaluate_relevance(self, feedback: str, gt_errors: List[str]) -> float:
        """Evaluate if feedback is relevant to the errors."""
        if not gt_errors:
            # No errors, feedback should be positive
            if "correct" in feedback.lower() or "good" in feedback.lower():
                return 5.0
            else:
                return 2.0
        
        # Check if feedback is relevant to the specific errors
        feedback_lower = feedback.lower()
        relevance_score = 0
        
        for error in gt_errors:
            if self._feedback_mentions_error(feedback_lower, error):
                relevance_score += 1
        
        # Normalize to 1-5 scale
        if relevance_score == len(gt_errors):
            return 5.0
        elif relevance_score > 0:
            return 3.0 + (relevance_score / len(gt_errors)) * 2.0
        else:
            return 1.0
    
    def _feedback_mentions_error(self, feedback: str, error: str) -> bool:
        """Check if feedback mentions a specific error type."""
        error_keywords = {
            "missing_force": ["missing", "force", "add force"],
            "wrong_direction": ["direction", "arrow", "pointing"],
            "extra_force": ["extra", "unnecessary", "remove"],
            "missing_ground": ["ground", "reference", "connection"],
            "wrong_polarity": ["polarity", "positive", "negative", "terminal"],
            "missing_component": ["component", "missing", "add"]
        }
        
        if error in error_keywords:
            keywords = error_keywords[error]
            return any(keyword in feedback for keyword in keywords)
        
        return False

class HallucinationDetector:
    """Detects hallucinations in model predictions."""
    
    def __init__(self):
        self.hallucination_indicators = [
            "nonexistent", "not visible", "cannot see", "imaginary",
            "phantom", "ghost", "invisible", "hidden"
        ]
    
    def detect_hallucinations(self, prediction: Dict, ground_truth: Dict) -> Dict[str, float]:
        """Detect hallucinations in predictions."""
        feedback = prediction.get('feedback', '')
        pred_errors = prediction.get('error_types', [])
        gt_errors = ground_truth.get('error_types', [])
        
        # Check for hallucination indicators in feedback
        hallucination_in_feedback = self._check_feedback_hallucinations(feedback)
        
        # Check for false positive errors (hallucinated errors)
        false_positives = len(set(pred_errors) - set(gt_errors))
        total_predictions = len(pred_errors)
        
        hallucination_rate = false_positives / total_predictions if total_predictions > 0 else 0
        false_positive_rate = false_positives / len(gt_errors) if len(gt_errors) > 0 else 0
        
        return {
            'hallucination_rate': hallucination_rate,
            'false_positive_rate': false_positive_rate,
            'hallucination_in_feedback': hallucination_in_feedback
        }
    
    def _check_feedback_hallucinations(self, feedback: str) -> bool:
        """Check if feedback contains hallucination indicators."""
        feedback_lower = feedback.lower()
        return any(indicator in feedback_lower for indicator in self.hallucination_indicators)

class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for Sketch2Feedback."""
    
    def __init__(self):
        self.error_detector = ErrorDetector()
        self.localization_evaluator = LocalizationEvaluator()
        self.feedback_evaluator = FeedbackEvaluator()
        self.hallucination_detector = HallucinationDetector()
    
    def evaluate_model(self, predictions: List[Dict], ground_truth: List[Dict]) -> EvaluationMetrics:
        """Evaluate a model comprehensively."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        # Initialize metrics
        error_metrics = []
        localization_metrics = []
        feedback_metrics = []
        hallucination_metrics = []
        
        # Evaluate each sample
        for pred, gt in zip(predictions, ground_truth):
            # Error detection
            error_metric = self.error_detector.detect_errors(pred, gt)
            error_metrics.append(error_metric)
            
            # Localization
            loc_metric = self.localization_evaluator.evaluate_localization(pred, gt)
            localization_metrics.append(loc_metric)
            
            # Feedback quality
            feedback_metric = self.feedback_evaluator.evaluate_feedback(pred, gt)
            feedback_metrics.append(feedback_metric)
            
            # Hallucination detection
            hallucination_metric = self.hallucination_detector.detect_hallucinations(pred, gt)
            hallucination_metrics.append(hallucination_metric)
        
        # Aggregate metrics
        return self._aggregate_metrics(error_metrics, localization_metrics, 
                                     feedback_metrics, hallucination_metrics)
    
    def _aggregate_metrics(self, error_metrics: List[Dict], localization_metrics: List[Dict],
                          feedback_metrics: List[Dict], hallucination_metrics: List[Dict]) -> EvaluationMetrics:
        """Aggregate metrics across all samples."""
        # Error detection metrics
        avg_precision = np.mean([m['precision'] for m in error_metrics])
        avg_recall = np.mean([m['recall'] for m in error_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in error_metrics])
        
        # Calculate accuracy
        total_tp = sum(m['true_positives'] for m in error_metrics)
        total_fp = sum(m['false_positives'] for m in error_metrics)
        total_fn = sum(m['false_negatives'] for m in error_metrics)
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
        # Localization metrics
        avg_iou_05 = np.mean([m['iou_at_05'] for m in localization_metrics])
        avg_iou_075 = np.mean([m['iou_at_075'] for m in localization_metrics])
        avg_mean_iou = np.mean([m['mean_iou'] for m in localization_metrics])
        
        # Feedback quality metrics
        avg_correctness = np.mean([m['correctness'] for m in feedback_metrics])
        avg_actionability = np.mean([m['actionability'] for m in feedback_metrics])
        avg_relevance = np.mean([m['relevance'] for m in feedback_metrics])
        
        # Hallucination metrics
        avg_hallucination_rate = np.mean([m['hallucination_rate'] for m in hallucination_metrics])
        avg_false_positive_rate = np.mean([m['false_positive_rate'] for m in hallucination_metrics])
        
        # Overall score (weighted combination)
        overall_score = (
            0.3 * avg_f1 +  # Error detection
            0.2 * avg_mean_iou +  # Localization
            0.3 * (avg_correctness + avg_actionability + avg_relevance) / 3 +  # Feedback quality
            0.2 * (1 - avg_hallucination_rate)  # Hallucination penalty
        )
        
        return EvaluationMetrics(
            f1_score=avg_f1,
            precision=avg_precision,
            recall=avg_recall,
            accuracy=accuracy,
            iou_at_05=avg_iou_05,
            iou_at_075=avg_iou_075,
            mean_iou=avg_mean_iou,
            feedback_correctness=avg_correctness,
            feedback_actionability=avg_actionability,
            feedback_relevance=avg_relevance,
            hallucination_rate=avg_hallucination_rate,
            false_positive_rate=avg_false_positive_rate,
            overall_score=overall_score
        )
    
    def generate_report(self, metrics: EvaluationMetrics, model_name: str) -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""
# Evaluation Report for {model_name}

## Error Detection Metrics
- **F1 Score**: {metrics.f1_score:.3f}
- **Precision**: {metrics.precision:.3f}
- **Recall**: {metrics.recall:.3f}
- **Accuracy**: {metrics.accuracy:.3f}

## Localization Metrics
- **IoU@0.5**: {metrics.iou_at_05:.3f}
- **IoU@0.75**: {metrics.iou_at_075:.3f}
- **Mean IoU**: {metrics.mean_iou:.3f}

## Feedback Quality Metrics
- **Correctness**: {metrics.feedback_correctness:.3f}
- **Actionability**: {metrics.feedback_actionability:.3f}
- **Relevance**: {metrics.feedback_relevance:.3f}

## Hallucination Metrics
- **Hallucination Rate**: {metrics.hallucination_rate:.3f}
- **False Positive Rate**: {metrics.false_positive_rate:.3f}

## Overall Score
- **Overall Score**: {metrics.overall_score:.3f}

## Interpretation
"""
        
        # Add interpretation
        if metrics.f1_score > 0.8:
            report += "- **Error Detection**: Excellent performance\n"
        elif metrics.f1_score > 0.6:
            report += "- **Error Detection**: Good performance\n"
        else:
            report += "- **Error Detection**: Needs improvement\n"
        
        if metrics.mean_iou > 0.7:
            report += "- **Localization**: Excellent performance\n"
        elif metrics.mean_iou > 0.5:
            report += "- **Localization**: Good performance\n"
        else:
            report += "- **Localization**: Needs improvement\n"
        
        if metrics.feedback_correctness > 4.0:
            report += "- **Feedback Quality**: Excellent performance\n"
        elif metrics.feedback_correctness > 3.0:
            report += "- **Feedback Quality**: Good performance\n"
        else:
            report += "- **Feedback Quality**: Needs improvement\n"
        
        if metrics.hallucination_rate < 0.1:
            report += "- **Hallucination**: Low hallucination rate\n"
        elif metrics.hallucination_rate < 0.3:
            report += "- **Hallucination**: Moderate hallucination rate\n"
        else:
            report += "- **Hallucination**: High hallucination rate\n"
        
        return report
    
    def plot_metrics(self, metrics: EvaluationMetrics, model_name: str, save_path: str = None):
        """Plot evaluation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Evaluation Metrics for {model_name}', fontsize=16)
        
        # Error detection metrics
        error_metrics = [metrics.f1_score, metrics.precision, metrics.recall, metrics.accuracy]
        error_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
        axes[0, 0].bar(error_labels, error_metrics, color='skyblue')
        axes[0, 0].set_title('Error Detection Metrics')
        axes[0, 0].set_ylim(0, 1)
        
        # Localization metrics
        loc_metrics = [metrics.iou_at_05, metrics.iou_at_075, metrics.mean_iou]
        loc_labels = ['IoU@0.5', 'IoU@0.75', 'Mean IoU']
        axes[0, 1].bar(loc_labels, loc_metrics, color='lightgreen')
        axes[0, 1].set_title('Localization Metrics')
        axes[0, 1].set_ylim(0, 1)
        
        # Feedback quality metrics
        feedback_metrics = [metrics.feedback_correctness, metrics.feedback_actionability, metrics.feedback_relevance]
        feedback_labels = ['Correctness', 'Actionability', 'Relevance']
        axes[1, 0].bar(feedback_labels, feedback_metrics, color='lightcoral')
        axes[1, 0].set_title('Feedback Quality Metrics')
        axes[1, 0].set_ylim(0, 5)
        
        # Hallucination metrics
        halluc_metrics = [metrics.hallucination_rate, metrics.false_positive_rate]
        halluc_labels = ['Hallucination Rate', 'False Positive Rate']
        axes[1, 1].bar(halluc_labels, halluc_metrics, color='orange')
        axes[1, 1].set_title('Hallucination Metrics')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def test_evaluation():
    """Test the evaluation suite."""
    # Create sample predictions and ground truth
    predictions = [
        {
            'error_types': ['missing_force'],
            'bounding_boxes': [[100, 100, 50, 50]],
            'feedback': 'Missing force detected. Add the normal force to your diagram.'
        },
        {
            'error_types': [],
            'bounding_boxes': [],
            'feedback': 'Great job! Your diagram looks correct.'
        }
    ]
    
    ground_truth = [
        {
            'error_types': ['missing_force'],
            'bounding_boxes': [[110, 110, 40, 40]]
        },
        {
            'error_types': [],
            'bounding_boxes': []
        }
    ]
    
    # Evaluate
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.evaluate_model(predictions, ground_truth)
    
    # Generate report
    report = evaluator.generate_report(metrics, "Test Model")
    print(report)
    
    return metrics

if __name__ == "__main__":
    test_evaluation()
