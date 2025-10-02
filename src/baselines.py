"""
Baseline models for Sketch2Feedback evaluation.
Includes end-to-end LMM, vision-only detector, and grammar-in-the-loop approaches.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

@dataclass
class Prediction:
    error_detected: bool
    error_type: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    feedback: str

class EndToEndLMM:
    """End-to-end Large Multimodal Model baseline."""
    
    def __init__(self, model_name: str = "microsoft/git-base"):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Simple feedback templates
        self.feedback_templates = {
            "missing_force": "I notice that some forces might be missing from your diagram. Make sure to include all forces acting on the object, such as weight, normal force, and any applied forces.",
            "wrong_direction": "Check the direction of your force arrows. Some forces appear to be pointing in the wrong direction.",
            "extra_force": "There might be an extra force in your diagram that shouldn't be there. Review which forces are actually acting on the object.",
            "missing_ground": "Your circuit is missing a ground connection. Most circuits need a ground reference point.",
            "wrong_polarity": "Check the polarity of your components. Make sure positive and negative terminals are correctly oriented.",
            "missing_component": "Some components appear to be missing from your circuit. Review the circuit requirements.",
            "correct": "Great job! Your diagram looks correct with no obvious errors."
        }
    
    def predict(self, image: Image.Image, diagram_type: str) -> Prediction:
        """Make prediction on a diagram."""
        try:
            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get image features
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
            
            # Simple heuristic-based error detection
            error_detected, error_type, confidence = self._detect_errors_heuristic(image, diagram_type)
            
            # Generate feedback
            feedback = self.feedback_templates.get(error_type, self.feedback_templates["correct"])
            
            # Estimate bounding box (simplified)
            bbox = self._estimate_error_bbox(image, error_type)
            
            return Prediction(
                error_detected=error_detected,
                error_type=error_type,
                bbox=bbox,
                confidence=confidence,
                feedback=feedback
            )
            
        except Exception as e:
            print(f"Error in EndToEndLMM prediction: {e}")
            return Prediction(
                error_detected=False,
                error_type="correct",
                bbox=(0, 0, 100, 100),
                confidence=0.5,
                feedback="Unable to analyze the diagram."
            )
    
    def _detect_errors_heuristic(self, image: Image.Image, diagram_type: str) -> Tuple[bool, str, float]:
        """Simple heuristic-based error detection."""
        # Convert to numpy array
        img_array = np.array(image)
        
        if diagram_type == "fbd":
            return self._detect_fbd_errors(img_array)
        elif diagram_type == "circuit":
            return self._detect_circuit_errors(img_array)
        else:
            return False, "correct", 0.5
    
    def _detect_fbd_errors(self, img_array: np.ndarray) -> Tuple[bool, str, float]:
        """Detect errors in free body diagrams."""
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect arrows (forces)
        arrows = self._detect_arrows(gray)
        
        # Detect objects (rectangles, circles)
        objects = self._detect_objects(gray)
        
        # Check for missing forces
        if len(objects) > 0 and len(arrows) == 0:
            return True, "missing_force", 0.8
        
        # Check for too many forces
        if len(arrows) > 5:
            return True, "extra_force", 0.6
        
        # Check for unbalanced forces (simplified)
        if len(arrows) > 0:
            total_force_x = 0
            total_force_y = 0
            
            for arrow in arrows:
                # Estimate force direction from arrow orientation
                dx = arrow[2] - arrow[0]
                dy = arrow[3] - arrow[1]
                total_force_x += dx
                total_force_y += dy
            
            # Check if forces are roughly balanced
            if abs(total_force_x) > 50 or abs(total_force_y) > 50:
                return True, "wrong_direction", 0.7
        
        return False, "correct", 0.9
    
    def _detect_circuit_errors(self, img_array: np.ndarray) -> Tuple[bool, str, float]:
        """Detect errors in circuit diagrams."""
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect components
        components = self._detect_circuit_components(gray)
        
        # Check for missing ground
        grounds = [comp for comp in components if comp[0] == "ground"]
        if len(components) > 0 and len(grounds) == 0:
            return True, "missing_ground", 0.8
        
        # Check for disconnected components
        if len(components) > 1:
            # Simple connectivity check
            if not self._check_connectivity(components):
                return True, "missing_component", 0.6
        
        return False, "correct", 0.9
    
    def _detect_arrows(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect arrow-like shapes."""
        arrows = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 1000:
                # Check if it looks like an arrow
                if self._is_arrow_shape(contour):
                    x, y, w, h = cv2.boundingRect(contour)
                    arrows.append((x, y, x+w, y+h))
        
        return arrows
    
    def _detect_objects(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular and circular objects."""
        objects = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                objects.append((x, y, x+w, y+h))
        
        return objects
    
    def _detect_circuit_components(self, gray: np.ndarray) -> List[Tuple[str, int, int, int, int]]:
        """Detect circuit components."""
        components = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Simple classification
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2:
                    comp_type = "resistor"
                elif aspect_ratio > 2:
                    comp_type = "battery"
                else:
                    comp_type = "component"
                
                components.append((comp_type, x, y, x+w, y+h))
        
        return components
    
    def _is_arrow_shape(self, contour) -> bool:
        """Check if contour resembles an arrow."""
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        solidity = contour_area / hull_area
        return solidity < 0.9
    
    def _check_connectivity(self, components: List[Tuple[str, int, int, int, int]]) -> bool:
        """Check if components are connected."""
        # Simplified connectivity check
        if len(components) < 2:
            return True
        
        # Check if components are close to each other
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                # Calculate distance between centers
                center1 = ((comp1[1] + comp1[3]) / 2, (comp1[2] + comp1[4]) / 2)
                center2 = ((comp2[1] + comp2[3]) / 2, (comp2[2] + comp2[4]) / 2)
                dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if dist < 100:  # Components are close
                    return True
        
        return False
    
    def _estimate_error_bbox(self, image: Image.Image, error_type: str) -> Tuple[int, int, int, int]:
        """Estimate bounding box for the error."""
        width, height = image.size
        
        if error_type == "missing_force":
            return (width//4, height//4, width//2, height//2)
        elif error_type == "wrong_direction":
            return (width//3, height//3, width//3, height//3)
        elif error_type == "extra_force":
            return (width//2, height//2, width//4, height//4)
        else:
            return (0, 0, width, height)

class VisionOnlyDetector:
    """Vision-only detector with static feedback."""
    
    def __init__(self):
        self.feedback_templates = {
            "missing_force": "Missing forces detected. Add all relevant forces to your diagram.",
            "wrong_direction": "Force direction error detected. Check arrow directions.",
            "extra_force": "Extra force detected. Remove unnecessary forces.",
            "missing_ground": "Missing ground connection. Add ground reference.",
            "wrong_polarity": "Polarity error detected. Check component orientation.",
            "missing_component": "Missing component detected. Add required components.",
            "correct": "No errors detected. Diagram appears correct."
        }
    
    def predict(self, image: Image.Image, diagram_type: str) -> Prediction:
        """Make prediction using only vision."""
        # Convert to numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect errors using computer vision
        error_detected, error_type, confidence = self._detect_errors_cv(gray, diagram_type)
        
        # Get static feedback
        feedback = self.feedback_templates.get(error_type, self.feedback_templates["correct"])
        
        # Estimate bounding box
        bbox = self._estimate_bbox(gray, error_type)
        
        return Prediction(
            error_detected=error_detected,
            error_type=error_type,
            bbox=bbox,
            confidence=confidence,
            feedback=feedback
        )
    
    def _detect_errors_cv(self, gray: np.ndarray, diagram_type: str) -> Tuple[bool, str, float]:
        """Detect errors using computer vision."""
        if diagram_type == "fbd":
            return self._detect_fbd_errors_cv(gray)
        elif diagram_type == "circuit":
            return self._detect_circuit_errors_cv(gray)
        else:
            return False, "correct", 0.5
    
    def _detect_fbd_errors_cv(self, gray: np.ndarray) -> Tuple[bool, str, float]:
        """Detect FBD errors using CV."""
        # Detect arrows
        arrows = self._detect_arrows_cv(gray)
        
        # Detect objects
        objects = self._detect_objects_cv(gray)
        
        # Check for missing forces
        if len(objects) > 0 and len(arrows) == 0:
            return True, "missing_force", 0.8
        
        # Check for too many forces
        if len(arrows) > 5:
            return True, "extra_force", 0.6
        
        return False, "correct", 0.9
    
    def _detect_circuit_errors_cv(self, gray: np.ndarray) -> Tuple[bool, str, float]:
        """Detect circuit errors using CV."""
        # Detect components
        components = self._detect_components_cv(gray)
        
        # Check for missing ground
        grounds = [comp for comp in components if comp[0] == "ground"]
        if len(components) > 0 and len(grounds) == 0:
            return True, "missing_ground", 0.8
        
        return False, "correct", 0.9
    
    def _detect_arrows_cv(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect arrows using CV."""
        arrows = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 1000:
                if self._is_arrow_cv(contour):
                    x, y, w, h = cv2.boundingRect(contour)
                    arrows.append((x, y, x+w, y+h))
        
        return arrows
    
    def _detect_objects_cv(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect objects using CV."""
        objects = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                objects.append((x, y, x+w, y+h))
        
        return objects
    
    def _detect_components_cv(self, gray: np.ndarray) -> List[Tuple[str, int, int, int, int]]:
        """Detect circuit components using CV."""
        components = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Simple classification
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2:
                    comp_type = "resistor"
                elif aspect_ratio > 2:
                    comp_type = "battery"
                else:
                    comp_type = "component"
                
                components.append((comp_type, x, y, x+w, y+h))
        
        return components
    
    def _is_arrow_cv(self, contour) -> bool:
        """Check if contour is an arrow using CV."""
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        solidity = contour_area / hull_area
        return solidity < 0.9
    
    def _estimate_bbox(self, gray: np.ndarray, error_type: str) -> Tuple[int, int, int, int]:
        """Estimate bounding box for error."""
        height, width = gray.shape
        
        if error_type == "missing_force":
            return (width//4, height//4, width//2, height//2)
        elif error_type == "wrong_direction":
            return (width//3, height//3, width//3, height//3)
        elif error_type == "extra_force":
            return (width//2, height//2, width//4, height//4)
        else:
            return (0, 0, width, height)

class BaselineEvaluator:
    """Evaluates baseline models."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_model(self, model, test_data: List[Dict], diagram_type: str) -> Dict[str, float]:
        """Evaluate a model on test data."""
        predictions = []
        ground_truth = []
        
        for sample in test_data:
            # Load image
            image = Image.open(sample['image_path'])
            
            # Make prediction
            pred = model.predict(image, diagram_type)
            predictions.append(pred)
            
            # Get ground truth
            gt_errors = sample.get('error_types', [])
            gt_has_error = len(gt_errors) > 0
            ground_truth.append(gt_has_error)
        
        # Calculate metrics
        pred_has_error = [p.error_detected for p in predictions]
        
        # Error detection metrics
        f1 = f1_score(ground_truth, pred_has_error, average='binary')
        precision = precision_score(ground_truth, pred_has_error, average='binary')
        recall = recall_score(ground_truth, pred_has_error, average='binary')
        
        # Feedback quality metrics (simplified)
        feedback_quality = self._evaluate_feedback_quality(predictions, test_data)
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'feedback_quality': feedback_quality
        }
    
    def _evaluate_feedback_quality(self, predictions: List[Prediction], test_data: List[Dict]) -> float:
        """Evaluate feedback quality (simplified)."""
        # This is a simplified evaluation - in practice, you'd use human raters or LLM-as-judge
        quality_scores = []
        
        for pred, sample in zip(predictions, test_data):
            # Check if feedback is relevant to the error
            gt_errors = sample.get('error_types', [])
            
            if pred.error_detected and len(gt_errors) > 0:
                # Feedback should be relevant
                if any(error in pred.feedback.lower() for error in gt_errors):
                    quality_scores.append(1.0)
                else:
                    quality_scores.append(0.5)
            elif not pred.error_detected and len(gt_errors) == 0:
                # No error detected, should be correct
                quality_scores.append(1.0)
            else:
                # Mismatch
                quality_scores.append(0.0)
        
        return np.mean(quality_scores) if quality_scores else 0.0

def test_baselines():
    """Test the baseline models."""
    # Create a simple test image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw a rectangle
    cv2.rectangle(image, (200, 150), (300, 250), (0, 0, 0), 2)
    
    # Draw an arrow
    cv2.arrowedLine(image, (250, 200), (250, 100), (0, 0, 255), 3)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Test end-to-end LMM
    print("Testing End-to-End LMM...")
    lmm_model = EndToEndLMM()
    lmm_pred = lmm_model.predict(pil_image, "fbd")
    print(f"LMM Prediction: {lmm_pred.error_detected}, {lmm_pred.error_type}")
    print(f"LMM Feedback: {lmm_pred.feedback}")
    
    # Test vision-only detector
    print("\nTesting Vision-Only Detector...")
    vision_model = VisionOnlyDetector()
    vision_pred = vision_model.predict(pil_image, "fbd")
    print(f"Vision Prediction: {vision_pred.error_detected}, {vision_pred.error_type}")
    print(f"Vision Feedback: {vision_pred.feedback}")
    
    return lmm_pred, vision_pred

if __name__ == "__main__":
    test_baselines()
