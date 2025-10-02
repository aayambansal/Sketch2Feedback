"""
Grammar-in-the-loop pipeline for Sketch2Feedback.
Detects primitives, converts to symbolic graph, checks constraints, and generates feedback.
"""

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class PrimitiveType(Enum):
    ARROW = "arrow"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    LINE = "line"
    TEXT = "text"
    BATTERY = "battery"
    RESISTOR = "resistor"
    DIODE = "diode"
    GROUND = "ground"

@dataclass
class Primitive:
    type: PrimitiveType
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[float, float]
    orientation: float = 0.0
    confidence: float = 1.0
    properties: Dict[str, Any] = None

@dataclass
class ConstraintViolation:
    constraint_type: str
    description: str
    severity: str  # "error", "warning"
    elements: List[str]

class PrimitiveDetector:
    """Detects basic primitives in diagrams using classical computer vision."""
    
    def __init__(self):
        self.min_contour_area = 50
        self.max_contour_area = 10000
        
    def detect_primitives(self, image: np.ndarray) -> List[Primitive]:
        """Detect all primitives in the image."""
        primitives = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect different primitive types
        primitives.extend(self._detect_arrows(gray))
        primitives.extend(self._detect_circles(gray))
        primitives.extend(self._detect_rectangles(gray))
        primitives.extend(self._detect_lines(gray))
        primitives.extend(self._detect_text(gray))
        primitives.extend(self._detect_circuit_components(gray))
        
        return primitives
    
    def _detect_arrows(self, gray: np.ndarray) -> List[Primitive]:
        """Detect arrow-like shapes using contour analysis."""
        arrows = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                # Check if contour looks like an arrow
                if self._is_arrow_shape(contour):
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w/2, y + h/2)
                    orientation = self._calculate_orientation(contour)
                    
                    arrows.append(Primitive(
                        type=PrimitiveType.ARROW,
                        bbox=(x, y, w, h),
                        center=center,
                        orientation=orientation,
                        confidence=0.8
                    ))
        
        return arrows
    
    def _detect_circles(self, gray: np.ndarray) -> List[Primitive]:
        """Detect circular shapes using HoughCircles."""
        circles = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        detected_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if detected_circles is not None:
            detected_circles = np.round(detected_circles[0, :]).astype("int")
            
            for (x, y, r) in detected_circles:
                bbox = (x - r, y - r, 2 * r, 2 * r)
                center = (x, y)
                
                circles.append(Primitive(
                    type=PrimitiveType.CIRCLE,
                    bbox=bbox,
                    center=center,
                    confidence=0.9
                ))
        
        return circles
    
    def _detect_rectangles(self, gray: np.ndarray) -> List[Primitive]:
        """Detect rectangular shapes using contour analysis."""
        rectangles = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w/2, y + h/2)
                    
                    rectangles.append(Primitive(
                        type=PrimitiveType.RECTANGLE,
                        bbox=(x, y, w, h),
                        center=center,
                        confidence=0.7
                    ))
        
        return rectangles
    
    def _detect_lines(self, gray: np.ndarray) -> List[Primitive]:
        """Detect line segments using HoughLinesP."""
        lines = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        detected_lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        
        if detected_lines is not None:
            for line in detected_lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate bounding box
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Calculate orientation
                orientation = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                lines.append(Primitive(
                    type=PrimitiveType.LINE,
                    bbox=bbox,
                    center=center,
                    orientation=orientation,
                    confidence=0.8
                ))
        
        return lines
    
    def _detect_text(self, gray: np.ndarray) -> List[Primitive]:
        """Detect text regions using MSER."""
        text_regions = []
        
        # Create MSER detector
        mser = cv2.MSER_create()
        
        # Detect regions
        regions, _ = mser.detectRegions(gray)
        
        for region in regions:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            
            if w > 10 and h > 10:  # Filter small regions
                center = (x + w/2, y + h/2)
                
                text_regions.append(Primitive(
                    type=PrimitiveType.TEXT,
                    bbox=(x, y, w, h),
                    center=center,
                    confidence=0.6
                ))
        
        return text_regions
    
    def _detect_circuit_components(self, gray: np.ndarray) -> List[Primitive]:
        """Detect circuit components using template matching."""
        components = []
        
        # This is a simplified version - in practice, you'd use more sophisticated methods
        # For now, we'll use contour analysis to detect potential components
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Component-sized objects
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w/2, y + h/2)
                
                # Simple heuristic to classify component type
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2:  # Square-ish
                    comp_type = PrimitiveType.RESISTOR
                elif aspect_ratio > 2:  # Wide
                    comp_type = PrimitiveType.BATTERY
                else:
                    comp_type = PrimitiveType.DIODE
                
                components.append(Primitive(
                    type=comp_type,
                    bbox=(x, y, w, h),
                    center=center,
                    confidence=0.5
                ))
        
        return components
    
    def _is_arrow_shape(self, contour) -> bool:
        """Check if contour resembles an arrow."""
        # Simplified arrow detection
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        # Arrow should have some concavity
        solidity = contour_area / hull_area
        return solidity < 0.9
    
    def _calculate_orientation(self, contour) -> float:
        """Calculate the orientation of a contour."""
        # Fit ellipse to get orientation
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            return ellipse[2]  # Angle
        return 0.0

class SymbolicGraph:
    """Represents the diagram as a symbolic graph."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.primitives = {}
        self.relationships = {}
    
    def add_primitive(self, primitive: Primitive, primitive_id: str):
        """Add a primitive to the graph."""
        self.primitives[primitive_id] = primitive
        self.graph.add_node(primitive_id, type=primitive.type.value)
    
    def add_relationship(self, from_id: str, to_id: str, relationship_type: str):
        """Add a relationship between primitives."""
        self.graph.add_edge(from_id, to_id, type=relationship_type)
    
    def get_connected_components(self) -> List[List[str]]:
        """Get connected components in the graph."""
        return list(nx.connected_components(self.graph))
    
    def get_neighbors(self, primitive_id: str) -> List[str]:
        """Get neighbors of a primitive."""
        return list(self.graph.neighbors(primitive_id))

class ConstraintChecker:
    """Checks domain-specific constraints on the symbolic graph."""
    
    def __init__(self):
        self.constraints = {
            'fbd': self._check_fbd_constraints,
            'circuit': self._check_circuit_constraints
        }
    
    def check_constraints(self, graph: SymbolicGraph, diagram_type: str) -> List[ConstraintViolation]:
        """Check all constraints for the given diagram type."""
        if diagram_type in self.constraints:
            return self.constraints[diagram_type](graph)
        return []
    
    def _check_fbd_constraints(self, graph: SymbolicGraph) -> List[ConstraintViolation]:
        """Check free body diagram constraints."""
        violations = []
        
        # Get all arrows (forces)
        arrows = [pid for pid, prim in graph.primitives.items() 
                 if prim.type == PrimitiveType.ARROW]
        
        # Get all objects (rectangles, circles)
        objects = [pid for pid, prim in graph.primitives.items() 
                  if prim.type in [PrimitiveType.RECTANGLE, PrimitiveType.CIRCLE]]
        
        # Check for missing forces
        if len(objects) > 0 and len(arrows) == 0:
            violations.append(ConstraintViolation(
                constraint_type="missing_forces",
                description="No forces detected on objects",
                severity="error",
                elements=objects
            ))
        
        # Check for unbalanced forces (simplified)
        if len(arrows) > 0:
            # Check if forces are roughly balanced (simplified heuristic)
            total_force_x = 0
            total_force_y = 0
            
            for arrow_id in arrows:
                arrow = graph.primitives[arrow_id]
                # Estimate force magnitude from arrow length
                force_mag = np.sqrt(arrow.bbox[2]**2 + arrow.bbox[3]**2)
                force_x = force_mag * np.cos(np.radians(arrow.orientation))
                force_y = force_mag * np.sin(np.radians(arrow.orientation))
                total_force_x += force_x
                total_force_y += force_y
            
            # Check if forces are roughly balanced
            if abs(total_force_x) > 50 or abs(total_force_y) > 50:
                violations.append(ConstraintViolation(
                    constraint_type="unbalanced_forces",
                    description="Forces appear unbalanced",
                    severity="warning",
                    elements=arrows
                ))
        
        return violations
    
    def _check_circuit_constraints(self, graph: SymbolicGraph) -> List[ConstraintViolation]:
        """Check circuit constraints."""
        violations = []
        
        # Get all components
        components = [pid for pid, prim in graph.primitives.items() 
                     if prim.type in [PrimitiveType.BATTERY, PrimitiveType.RESISTOR, 
                                     PrimitiveType.DIODE, PrimitiveType.GROUND]]
        
        # Check for missing ground
        grounds = [pid for pid, prim in graph.primitives.items() 
                  if prim.type == PrimitiveType.GROUND]
        
        if len(components) > 0 and len(grounds) == 0:
            violations.append(ConstraintViolation(
                constraint_type="missing_ground",
                description="No ground connection found",
                severity="error",
                elements=components
            ))
        
        # Check for disconnected components
        connected_components = graph.get_connected_components()
        if len(connected_components) > 1:
            violations.append(ConstraintViolation(
                constraint_type="disconnected_components",
                description="Some components are not connected",
                severity="warning",
                elements=[comp for cc in connected_components for comp in cc]
            ))
        
        # Check for polarity issues (simplified)
        batteries = [pid for pid, prim in graph.primitives.items() 
                    if prim.type == PrimitiveType.BATTERY]
        diodes = [pid for pid, prim in graph.primitives.items() 
                 if prim.type == PrimitiveType.DIODE]
        
        # This is a simplified check - in practice, you'd analyze the actual polarity
        if len(batteries) > 0 or len(diodes) > 0:
            # Check if components are properly oriented
            for comp_id in batteries + diodes:
                comp = graph.primitives[comp_id]
                # Simple heuristic: check if component is roughly horizontal/vertical
                if abs(comp.orientation) > 45 and abs(comp.orientation) < 135:
                    violations.append(ConstraintViolation(
                        constraint_type="polarity_issue",
                        description="Component may have incorrect polarity",
                        severity="warning",
                        elements=[comp_id]
                    ))
        
        return violations

class FeedbackGenerator:
    """Generates rubric-aligned feedback using a small language model."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_feedback(self, violations: List[ConstraintViolation], 
                         diagram_type: str, primitives: Dict[str, Primitive]) -> str:
        """Generate feedback based on constraint violations."""
        
        if not violations:
            return "Great job! Your diagram appears to be correct with no obvious errors."
        
        feedback_parts = []
        
        for violation in violations:
            if violation.severity == "error":
                feedback_parts.append(self._generate_error_feedback(violation, diagram_type))
            else:
                feedback_parts.append(self._generate_warning_feedback(violation, diagram_type))
        
        # Add general suggestions
        feedback_parts.append(self._generate_general_suggestions(diagram_type))
        
        return " ".join(feedback_parts)
    
    def _generate_error_feedback(self, violation: ConstraintViolation, diagram_type: str) -> str:
        """Generate feedback for errors."""
        if violation.constraint_type == "missing_forces":
            return "Error: No forces are shown on the object. Remember to draw all forces acting on the object, including weight, normal force, and any applied forces."
        elif violation.constraint_type == "missing_ground":
            return "Error: No ground connection found. Circuits typically need a ground reference point."
        elif violation.constraint_type == "unbalanced_forces":
            return "Error: The forces appear unbalanced. Check that all forces are correctly drawn and labeled."
        else:
            return f"Error detected: {violation.description}"
    
    def _generate_warning_feedback(self, violation: ConstraintViolation, diagram_type: str) -> str:
        """Generate feedback for warnings."""
        if violation.constraint_type == "disconnected_components":
            return "Warning: Some components appear disconnected. Make sure all components are properly connected in the circuit."
        elif violation.constraint_type == "polarity_issue":
            return "Warning: Check the polarity of components like batteries and diodes. Make sure positive and negative terminals are correctly oriented."
        else:
            return f"Warning: {violation.description}"
    
    def _generate_general_suggestions(self, diagram_type: str) -> str:
        """Generate general suggestions based on diagram type."""
        if diagram_type == "fbd":
            return "General tips: Make sure to draw forces as arrows with clear labels, and ensure all forces acting on the object are included."
        elif diagram_type == "circuit":
            return "General tips: Ensure all components are properly connected, include a ground reference, and check component polarities."
        else:
            return "Double-check your diagram for accuracy and completeness."

class GrammarInTheLoopPipeline:
    """Main pipeline that orchestrates the grammar-in-the-loop approach."""
    
    def __init__(self):
        self.detector = PrimitiveDetector()
        self.constraint_checker = ConstraintChecker()
        self.feedback_generator = FeedbackGenerator()
    
    def process_diagram(self, image: np.ndarray, diagram_type: str) -> Dict[str, Any]:
        """Process a diagram through the grammar-in-the-loop pipeline."""
        
        # Step 1: Detect primitives
        primitives = self.detector.detect_primitives(image)
        
        # Step 2: Build symbolic graph
        graph = SymbolicGraph()
        for i, primitive in enumerate(primitives):
            graph.add_primitive(primitive, f"prim_{i}")
        
        # Step 3: Add relationships based on spatial proximity
        self._add_spatial_relationships(graph)
        
        # Step 4: Check constraints
        violations = self.constraint_checker.check_constraints(graph, diagram_type)
        
        # Step 5: Generate feedback
        feedback = self.feedback_generator.generate_feedback(violations, diagram_type, graph.primitives)
        
        return {
            'primitives': primitives,
            'violations': violations,
            'feedback': feedback,
            'graph': graph
        }
    
    def _add_spatial_relationships(self, graph: SymbolicGraph):
        """Add relationships based on spatial proximity."""
        primitive_ids = list(graph.primitives.keys())
        
        for i, id1 in enumerate(primitive_ids):
            for j, id2 in enumerate(primitive_ids[i+1:], i+1):
                prim1 = graph.primitives[id1]
                prim2 = graph.primitives[id2]
                
                # Calculate distance between centers
                dist = np.sqrt((prim1.center[0] - prim2.center[0])**2 + 
                              (prim1.center[1] - prim2.center[1])**2)
                
                # Add relationship if close enough
                if dist < 100:  # Threshold for proximity
                    graph.add_relationship(id1, id2, "nearby")

def test_pipeline():
    """Test the grammar-in-the-loop pipeline."""
    # Create a simple test image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw a simple rectangle
    cv2.rectangle(image, (200, 150), (300, 250), (0, 0, 0), 2)
    
    # Draw an arrow (force)
    cv2.arrowedLine(image, (250, 200), (250, 100), (0, 0, 255), 3)
    
    # Initialize pipeline
    pipeline = GrammarInTheLoopPipeline()
    
    # Process the image
    result = pipeline.process_diagram(image, "fbd")
    
    print("Detected primitives:", len(result['primitives']))
    print("Constraint violations:", len(result['violations']))
    print("Generated feedback:", result['feedback'])
    
    return result

if __name__ == "__main__":
    test_pipeline()
