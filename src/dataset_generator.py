"""
Dataset generator for Sketch2Feedback project.
Generates synthetic FBD-10 and Circuit-10 datasets with controllable errors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import cv2
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random

class ErrorType(Enum):
    MISSING_FORCE = "missing_force"
    WRONG_DIRECTION = "wrong_direction"
    EXTRA_FORCE = "extra_force"
    WRONG_LABEL = "wrong_label"
    MISSING_GROUND = "missing_ground"
    WRONG_POLARITY = "wrong_polarity"
    WRONG_CONNECTION = "wrong_connection"
    MISSING_COMPONENT = "missing_component"

@dataclass
class ForceVector:
    x: float
    y: float
    magnitude: float
    direction: float  # in degrees
    label: str
    color: str = 'red'
    error_type: ErrorType = None

@dataclass
class CircuitComponent:
    component_type: str  # 'resistor', 'battery', 'diode', 'ground'
    x: float
    y: float
    orientation: float
    label: str
    polarity: str = None  # for battery/diode
    error_type: ErrorType = None

class FBDGenerator:
    """Generates Free Body Diagrams with controllable errors."""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.scenarios = [
            "block_on_inclined_plane",
            "hanging_mass",
            "pushing_block",
            "car_on_road",
            "pendulum",
            "block_on_table",
            "sliding_block",
            "elevator",
            "projectile",
            "spring_mass"
        ]
    
    def _serialize_forces(self, forces: List[ForceVector]) -> List[Dict]:
        """Convert ForceVector objects to JSON-serializable format."""
        return [{'x': f.x, 'y': f.y, 'magnitude': f.magnitude, 'direction': f.direction, 
                'label': f.label, 'color': f.color, 'error_type': f.error_type.value if f.error_type else None} 
                for f in forces]
    
    def generate_scenario(self, scenario_name: str, error_types: List[ErrorType] = None) -> Dict[str, Any]:
        """Generate a specific physics scenario with errors."""
        if error_types is None:
            error_types = []
        
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Generate base scenario
        if scenario_name == "block_on_inclined_plane":
            return self._generate_inclined_plane(ax, error_types)
        elif scenario_name == "hanging_mass":
            return self._generate_hanging_mass(ax, error_types)
        elif scenario_name == "pushing_block":
            return self._generate_pushing_block(ax, error_types)
        elif scenario_name == "car_on_road":
            return self._generate_car_on_road(ax, error_types)
        elif scenario_name == "pendulum":
            return self._generate_pendulum(ax, error_types)
        elif scenario_name == "block_on_table":
            return self._generate_block_on_table(ax, error_types)
        elif scenario_name == "sliding_block":
            return self._generate_sliding_block(ax, error_types)
        elif scenario_name == "elevator":
            return self._generate_elevator(ax, error_types)
        elif scenario_name == "projectile":
            return self._generate_projectile(ax, error_types)
        elif scenario_name == "spring_mass":
            return self._generate_spring_mass(ax, error_types)
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
    
    def _generate_inclined_plane(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate block on inclined plane scenario."""
        # Draw inclined plane
        plane_x = [100, 400]
        plane_y = [200, 350]
        ax.plot(plane_x, plane_y, 'k-', linewidth=3)
        
        # Draw block
        block_x, block_y = 250, 275
        block = Rectangle((block_x-20, block_y-10), 40, 20, 
                         facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(block)
        
        # Generate forces
        forces = []
        annotations = []
        
        # Weight (always present)
        weight = ForceVector(block_x, block_y-10, 50, 270, "mg")
        forces.append(weight)
        
        # Normal force
        if ErrorType.MISSING_FORCE not in error_types:
            normal = ForceVector(block_x+20, block_y, 30, 45, "N")
            forces.append(normal)
        elif ErrorType.WRONG_DIRECTION in error_types:
            normal = ForceVector(block_x+20, block_y, 30, 90, "N")  # Wrong direction
            normal.error_type = ErrorType.WRONG_DIRECTION
            forces.append(normal)
        
        # Friction
        if ErrorType.MISSING_FORCE not in error_types:
            friction = ForceVector(block_x-20, block_y, 20, 225, "f")
            forces.append(friction)
        elif ErrorType.WRONG_DIRECTION in error_types:
            friction = ForceVector(block_x-20, block_y, 20, 45, "f")  # Wrong direction
            friction.error_type = ErrorType.WRONG_DIRECTION
            forces.append(friction)
        
        # Extra force
        if ErrorType.EXTRA_FORCE in error_types:
            extra = ForceVector(block_x, block_y+10, 25, 0, "F_applied")
            extra.error_type = ErrorType.EXTRA_FORCE
            forces.append(extra)
        
        # Draw forces
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        # Add scenario annotation
        annotations.append({
            'type': 'scenario',
            'bbox': [50, 50, 100, 30],
            'label': 'Block on Inclined Plane',
            'error_type': None
        })
        
        return {
            'scenario': 'block_on_inclined_plane',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_hanging_mass(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate hanging mass scenario."""
        # Draw ceiling
        ax.plot([200, 400], [500, 500], 'k-', linewidth=3)
        
        # Draw rope
        ax.plot([300, 300], [500, 300], 'k-', linewidth=2)
        
        # Draw mass
        mass = Circle((300, 280), 20, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(mass)
        
        forces = []
        annotations = []
        
        # Tension
        if ErrorType.MISSING_FORCE not in error_types:
            tension = ForceVector(300, 300, 40, 90, "T")
            forces.append(tension)
        
        # Weight
        weight = ForceVector(300, 260, 40, 270, "mg")
        forces.append(weight)
        
        # Extra horizontal force
        if ErrorType.EXTRA_FORCE in error_types:
            extra = ForceVector(320, 280, 30, 0, "F_wind")
            extra.error_type = ErrorType.EXTRA_FORCE
            forces.append(extra)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'hanging_mass',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_pushing_block(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate pushing block scenario."""
        # Draw ground
        ax.plot([100, 500], [200, 200], 'k-', linewidth=3)
        
        # Draw block
        block = Rectangle((300, 200), 60, 40, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(block)
        
        forces = []
        annotations = []
        
        # Applied force
        applied = ForceVector(300, 220, 50, 0, "F_applied")
        forces.append(applied)
        
        # Weight
        weight = ForceVector(330, 200, 40, 270, "mg")
        forces.append(weight)
        
        # Normal force
        if ErrorType.MISSING_FORCE not in error_types:
            normal = ForceVector(330, 240, 40, 90, "N")
            forces.append(normal)
        
        # Friction
        if ErrorType.MISSING_FORCE not in error_types:
            friction = ForceVector(300, 220, 30, 180, "f")
            forces.append(friction)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'pushing_block',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _draw_force_vector(self, ax, force: ForceVector):
        """Draw a force vector with arrow."""
        # Calculate arrow end point
        dx = force.magnitude * np.cos(np.radians(force.direction))
        dy = force.magnitude * np.sin(np.radians(force.direction))
        
        # Draw arrow
        arrow = patches.FancyArrowPatch(
            (force.x, force.y), (force.x + dx, force.y + dy),
            arrowstyle='->', mutation_scale=20, color=force.color, linewidth=2
        )
        ax.add_patch(arrow)
        
        # Add label
        ax.text(force.x + dx + 5, force.y + dy + 5, force.label, 
                fontsize=12, ha='left', va='bottom')
    
    def _generate_car_on_road(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate car on road scenario."""
        # Draw road
        ax.plot([100, 500], [300, 300], 'k-', linewidth=4)
        
        # Draw car
        car = Rectangle((250, 300), 80, 40, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(car)
        
        forces = []
        annotations = []
        
        # Engine force
        engine = ForceVector(250, 320, 60, 0, "F_engine")
        forces.append(engine)
        
        # Weight
        weight = ForceVector(290, 300, 50, 270, "mg")
        forces.append(weight)
        
        # Normal forces
        if ErrorType.MISSING_FORCE not in error_types:
            normal_front = ForceVector(270, 340, 25, 90, "N_front")
            normal_rear = ForceVector(310, 340, 25, 90, "N_rear")
            forces.extend([normal_front, normal_rear])
        
        # Air resistance
        if ErrorType.MISSING_FORCE not in error_types:
            air_resistance = ForceVector(330, 320, 20, 180, "F_air")
            forces.append(air_resistance)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'car_on_road',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_pendulum(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate pendulum scenario."""
        # Draw support
        ax.plot([300, 300], [500, 450], 'k-', linewidth=3)
        
        # Draw pendulum
        ax.plot([300, 300], [450, 300], 'k-', linewidth=2)
        
        # Draw mass
        mass = Circle((300, 280), 15, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(mass)
        
        forces = []
        annotations = []
        
        # Tension
        if ErrorType.MISSING_FORCE not in error_types:
            tension = ForceVector(300, 300, 40, 90, "T")
            forces.append(tension)
        
        # Weight
        weight = ForceVector(300, 265, 40, 270, "mg")
        forces.append(weight)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'pendulum',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_block_on_table(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate block on table scenario."""
        # Draw table
        table = Rectangle((150, 200), 200, 20, facecolor='brown', edgecolor='black', linewidth=2)
        ax.add_patch(table)
        
        # Draw block
        block = Rectangle((250, 220), 40, 30, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(block)
        
        forces = []
        annotations = []
        
        # Weight
        weight = ForceVector(270, 220, 40, 270, "mg")
        forces.append(weight)
        
        # Normal force
        if ErrorType.MISSING_FORCE not in error_types:
            normal = ForceVector(270, 250, 40, 90, "N")
            forces.append(normal)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'block_on_table',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_sliding_block(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate sliding block scenario."""
        # Draw surface
        ax.plot([100, 500], [250, 250], 'k-', linewidth=3)
        
        # Draw block
        block = Rectangle((300, 250), 50, 30, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(block)
        
        forces = []
        annotations = []
        
        # Weight
        weight = ForceVector(325, 250, 40, 270, "mg")
        forces.append(weight)
        
        # Normal force
        if ErrorType.MISSING_FORCE not in error_types:
            normal = ForceVector(325, 280, 40, 90, "N")
            forces.append(normal)
        
        # Friction
        if ErrorType.MISSING_FORCE not in error_types:
            friction = ForceVector(300, 265, 30, 180, "f")
            forces.append(friction)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'sliding_block',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_elevator(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate elevator scenario."""
        # Draw elevator shaft
        ax.plot([200, 200], [100, 400], 'k-', linewidth=3)
        ax.plot([400, 400], [100, 400], 'k-', linewidth=3)
        
        # Draw elevator
        elevator = Rectangle((200, 200), 200, 100, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(elevator)
        
        forces = []
        annotations = []
        
        # Tension
        if ErrorType.MISSING_FORCE not in error_types:
            tension = ForceVector(300, 300, 50, 90, "T")
            forces.append(tension)
        
        # Weight
        weight = ForceVector(300, 200, 50, 270, "mg")
        forces.append(weight)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'elevator',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_projectile(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate projectile scenario."""
        # Draw ground
        ax.plot([100, 500], [200, 200], 'k-', linewidth=3)
        
        # Draw projectile
        projectile = Circle((300, 300), 8, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(projectile)
        
        forces = []
        annotations = []
        
        # Weight
        weight = ForceVector(300, 292, 30, 270, "mg")
        forces.append(weight)
        
        # Air resistance
        if ErrorType.MISSING_FORCE not in error_types:
            air_resistance = ForceVector(308, 300, 20, 180, "F_air")
            forces.append(air_resistance)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'projectile',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_spring_mass(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate spring-mass scenario."""
        # Draw wall
        ax.plot([100, 100], [200, 400], 'k-', linewidth=4)
        
        # Draw spring
        spring_x = np.linspace(100, 250, 20)
        spring_y = 300 + 10 * np.sin(spring_x * 0.3)
        ax.plot(spring_x, spring_y, 'k-', linewidth=2)
        
        # Draw mass
        mass = Rectangle((250, 280), 40, 40, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(mass)
        
        forces = []
        annotations = []
        
        # Spring force
        if ErrorType.MISSING_FORCE not in error_types:
            spring_force = ForceVector(250, 300, 40, 180, "F_spring")
            forces.append(spring_force)
        
        # Weight
        weight = ForceVector(270, 280, 40, 270, "mg")
        forces.append(weight)
        
        for force in forces:
            self._draw_force_vector(ax, force)
            annotations.append({
                'type': 'force',
                'bbox': [force.x-10, force.y-10, 20, 20],
                'label': force.label,
                'error_type': force.error_type.value if force.error_type else None
            })
        
        return {
            'scenario': 'spring_mass',
            'forces': self._serialize_forces(forces),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }

class CircuitGenerator:
    """Generates simple DC circuits with controllable errors."""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.scenarios = [
            "simple_series",
            "simple_parallel", 
            "series_parallel",
            "diode_circuit",
            "battery_resistor",
            "led_circuit",
            "voltage_divider",
            "current_source",
            "capacitor_circuit",
            "grounded_circuit"
        ]
    
    def _serialize_components(self, components: List[CircuitComponent]) -> List[Dict]:
        """Convert CircuitComponent objects to JSON-serializable format."""
        return [{'component_type': c.component_type, 'x': c.x, 'y': c.y, 'orientation': c.orientation,
                'label': c.label, 'polarity': c.polarity, 'error_type': c.error_type.value if c.error_type else None}
                for c in components]
    
    def generate_scenario(self, scenario_name: str, error_types: List[ErrorType] = None) -> Dict[str, Any]:
        """Generate a specific circuit scenario with errors."""
        if error_types is None:
            error_types = []
        
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Generate base scenario
        if scenario_name == "simple_series":
            return self._generate_simple_series(ax, error_types)
        elif scenario_name == "simple_parallel":
            return self._generate_simple_parallel(ax, error_types)
        elif scenario_name == "series_parallel":
            return self._generate_series_parallel(ax, error_types)
        elif scenario_name == "diode_circuit":
            return self._generate_diode_circuit(ax, error_types)
        elif scenario_name == "battery_resistor":
            return self._generate_battery_resistor(ax, error_types)
        elif scenario_name == "led_circuit":
            return self._generate_led_circuit(ax, error_types)
        elif scenario_name == "voltage_divider":
            return self._generate_voltage_divider(ax, error_types)
        elif scenario_name == "current_source":
            return self._generate_current_source(ax, error_types)
        elif scenario_name == "capacitor_circuit":
            return self._generate_capacitor_circuit(ax, error_types)
        elif scenario_name == "grounded_circuit":
            return self._generate_grounded_circuit(ax, error_types)
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
    
    def _generate_simple_series(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate simple series circuit."""
        components = []
        annotations = []
        
        # Battery
        battery = CircuitComponent('battery', 150, 300, 0, 'V1')
        if ErrorType.WRONG_POLARITY in error_types:
            battery.polarity = 'reversed'
            battery.error_type = ErrorType.WRONG_POLARITY
        components.append(battery)
        
        # Resistor 1
        resistor1 = CircuitComponent('resistor', 300, 300, 0, 'R1')
        components.append(resistor1)
        
        # Resistor 2
        if ErrorType.MISSING_COMPONENT not in error_types:
            resistor2 = CircuitComponent('resistor', 450, 300, 0, 'R2')
            components.append(resistor2)
        
        # Ground
        if ErrorType.MISSING_GROUND not in error_types:
            ground = CircuitComponent('ground', 600, 300, 0, 'GND')
            components.append(ground)
        elif ErrorType.MISSING_GROUND in error_types:
            # Missing ground is an error
            pass
        
        # Draw components and connections
        self._draw_circuit(ax, components)
        
        for comp in components:
            annotations.append({
                'type': 'component',
                'bbox': [comp.x-15, comp.y-15, 30, 30],
                'label': comp.label,
                'error_type': comp.error_type.value if comp.error_type else None
            })
        
        return {
            'scenario': 'simple_series',
            'components': self._serialize_components(components),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_simple_parallel(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate simple parallel circuit."""
        components = []
        annotations = []
        
        # Battery
        battery = CircuitComponent('battery', 200, 400, 0, 'V1')
        components.append(battery)
        
        # Resistor 1 (top branch)
        resistor1 = CircuitComponent('resistor', 350, 450, 0, 'R1')
        components.append(resistor1)
        
        # Resistor 2 (bottom branch)
        if ErrorType.MISSING_COMPONENT not in error_types:
            resistor2 = CircuitComponent('resistor', 350, 350, 0, 'R2')
            components.append(resistor2)
        
        # Ground
        if ErrorType.MISSING_GROUND not in error_types:
            ground = CircuitComponent('ground', 500, 400, 0, 'GND')
            components.append(ground)
        
        self._draw_circuit(ax, components)
        
        for comp in components:
            annotations.append({
                'type': 'component',
                'bbox': [comp.x-15, comp.y-15, 30, 30],
                'label': comp.label,
                'error_type': comp.error_type.value if comp.error_type else None
            })
        
        return {
            'scenario': 'simple_parallel',
            'components': self._serialize_components(components),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _generate_diode_circuit(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate diode circuit."""
        components = []
        annotations = []
        
        # Battery
        battery = CircuitComponent('battery', 150, 300, 0, 'V1')
        components.append(battery)
        
        # Diode
        diode = CircuitComponent('diode', 300, 300, 0, 'D1')
        if ErrorType.WRONG_POLARITY in error_types:
            diode.polarity = 'reversed'
            diode.error_type = ErrorType.WRONG_POLARITY
        components.append(diode)
        
        # Resistor
        resistor = CircuitComponent('resistor', 450, 300, 0, 'R1')
        components.append(resistor)
        
        # Ground
        if ErrorType.MISSING_GROUND not in error_types:
            ground = CircuitComponent('ground', 600, 300, 0, 'GND')
            components.append(ground)
        
        self._draw_circuit(ax, components)
        
        for comp in components:
            annotations.append({
                'type': 'component',
                'bbox': [comp.x-15, comp.y-15, 30, 30],
                'label': comp.label,
                'error_type': comp.error_type.value if comp.error_type else None
            })
        
        return {
            'scenario': 'diode_circuit',
            'components': self._serialize_components(components),
            'annotations': annotations,
            'error_types': [e.value for e in error_types]
        }
    
    def _draw_circuit(self, ax, components: List[CircuitComponent]):
        """Draw circuit components and connections."""
        # Draw connections first
        if len(components) > 1:
            for i in range(len(components) - 1):
                ax.plot([components[i].x, components[i+1].x], 
                       [components[i].y, components[i+1].y], 'k-', linewidth=2)
        
        # Draw components
        for comp in components:
            if comp.component_type == 'battery':
                self._draw_battery(ax, comp)
            elif comp.component_type == 'resistor':
                self._draw_resistor(ax, comp)
            elif comp.component_type == 'diode':
                self._draw_diode(ax, comp)
            elif comp.component_type == 'ground':
                self._draw_ground(ax, comp)
    
    def _draw_battery(self, ax, comp: CircuitComponent):
        """Draw battery symbol."""
        # Battery body
        battery_rect = Rectangle((comp.x-10, comp.y-5), 20, 10, 
                               facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(battery_rect)
        
        # Positive terminal
        ax.plot([comp.x+10, comp.x+10], [comp.y-2, comp.y+2], 'k-', linewidth=3)
        
        # Negative terminal
        ax.plot([comp.x-10, comp.x-10], [comp.y-2, comp.y+2], 'k-', linewidth=3)
        
        # Label
        ax.text(comp.x, comp.y-20, comp.label, ha='center', va='top', fontsize=10)
        
        # Polarity indicators
        if comp.polarity == 'reversed':
            ax.text(comp.x+10, comp.y+15, '-', ha='center', va='bottom', fontsize=12, color='red')
            ax.text(comp.x-10, comp.y+15, '+', ha='center', va='bottom', fontsize=12, color='red')
        else:
            ax.text(comp.x+10, comp.y+15, '+', ha='center', va='bottom', fontsize=12)
            ax.text(comp.x-10, comp.y+15, '-', ha='center', va='bottom', fontsize=12)
    
    def _draw_resistor(self, ax, comp: CircuitComponent):
        """Draw resistor symbol."""
        # Resistor body
        resistor_rect = Rectangle((comp.x-8, comp.y-3), 16, 6, 
                                facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(resistor_rect)
        
        # Label
        ax.text(comp.x, comp.y-20, comp.label, ha='center', va='top', fontsize=10)
    
    def _draw_diode(self, ax, comp: CircuitComponent):
        """Draw diode symbol."""
        # Diode triangle
        triangle = patches.Polygon([(comp.x-5, comp.y-5), (comp.x+5, comp.y), (comp.x-5, comp.y+5)],
                                 facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        # Diode line
        ax.plot([comp.x+5, comp.x+10], [comp.y, comp.y], 'k-', linewidth=2)
        
        # Label
        ax.text(comp.x, comp.y-20, comp.label, ha='center', va='top', fontsize=10)
        
        # Polarity indicators
        if comp.polarity == 'reversed':
            ax.text(comp.x-5, comp.y+15, 'A', ha='center', va='bottom', fontsize=10, color='red')
            ax.text(comp.x+10, comp.y+15, 'K', ha='center', va='bottom', fontsize=10, color='red')
        else:
            ax.text(comp.x-5, comp.y+15, 'A', ha='center', va='bottom', fontsize=10)
            ax.text(comp.x+10, comp.y+15, 'K', ha='center', va='bottom', fontsize=10)
    
    def _draw_ground(self, ax, comp: CircuitComponent):
        """Draw ground symbol."""
        # Ground line
        ax.plot([comp.x-10, comp.x+10], [comp.y, comp.y], 'k-', linewidth=2)
        
        # Ground vertical lines
        ax.plot([comp.x-5, comp.x-5], [comp.y, comp.y+5], 'k-', linewidth=2)
        ax.plot([comp.x, comp.x], [comp.y, comp.y+5], 'k-', linewidth=2)
        ax.plot([comp.x+5, comp.x+5], [comp.y, comp.y+5], 'k-', linewidth=2)
        
        # Label
        ax.text(comp.x, comp.y-20, comp.label, ha='center', va='top', fontsize=10)
    
    # Additional circuit generation methods would go here...
    def _generate_series_parallel(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate series-parallel circuit."""
        result = self._generate_simple_series(ax, error_types)  # Simplified for now
        result['scenario'] = 'series_parallel'
        return result
    
    def _generate_battery_resistor(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate battery-resistor circuit."""
        result = self._generate_simple_series(ax, error_types)  # Simplified for now
        result['scenario'] = 'battery_resistor'
        return result
    
    def _generate_led_circuit(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate LED circuit."""
        result = self._generate_diode_circuit(ax, error_types)  # Simplified for now
        result['scenario'] = 'led_circuit'
        return result
    
    def _generate_voltage_divider(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate voltage divider circuit."""
        result = self._generate_simple_series(ax, error_types)  # Simplified for now
        result['scenario'] = 'voltage_divider'
        return result
    
    def _generate_current_source(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate current source circuit."""
        result = self._generate_simple_series(ax, error_types)  # Simplified for now
        result['scenario'] = 'current_source'
        return result
    
    def _generate_capacitor_circuit(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate capacitor circuit."""
        result = self._generate_simple_series(ax, error_types)  # Simplified for now
        result['scenario'] = 'capacitor_circuit'
        return result
    
    def _generate_grounded_circuit(self, ax, error_types: List[ErrorType]) -> Dict[str, Any]:
        """Generate grounded circuit."""
        result = self._generate_simple_series(ax, error_types)  # Simplified for now
        result['scenario'] = 'grounded_circuit'
        return result

def generate_fbd_dataset(num_samples_per_scenario: int = 10, output_dir: str = "data/fbd_10"):
    """Generate FBD-10 dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = FBDGenerator()
    error_combinations = [
        [],  # No errors
        [ErrorType.MISSING_FORCE],
        [ErrorType.WRONG_DIRECTION],
        [ErrorType.EXTRA_FORCE],
        [ErrorType.MISSING_FORCE, ErrorType.WRONG_DIRECTION],
        [ErrorType.EXTRA_FORCE, ErrorType.WRONG_DIRECTION],
    ]
    
    dataset = []
    
    for scenario in generator.scenarios:
        for i in range(num_samples_per_scenario):
            error_types = random.choice(error_combinations)
            data = generator.generate_scenario(scenario, error_types)
            
            # Save image
            plt.savefig(f"{output_dir}/{scenario}_{i:02d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save metadata
            data['image_path'] = f"{scenario}_{i:02d}.png"
            dataset.append(data)
    
    # Save dataset metadata
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return dataset

def generate_circuit_dataset(num_samples_per_scenario: int = 10, output_dir: str = "data/circuit_10"):
    """Generate Circuit-10 dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = CircuitGenerator()
    error_combinations = [
        [],  # No errors
        [ErrorType.MISSING_GROUND],
        [ErrorType.WRONG_POLARITY],
        [ErrorType.MISSING_COMPONENT],
        [ErrorType.MISSING_GROUND, ErrorType.WRONG_POLARITY],
        [ErrorType.MISSING_COMPONENT, ErrorType.WRONG_POLARITY],
    ]
    
    dataset = []
    
    for scenario in generator.scenarios:
        for i in range(num_samples_per_scenario):
            error_types = random.choice(error_combinations)
            data = generator.generate_scenario(scenario, error_types)
            
            # Save image
            plt.savefig(f"{output_dir}/{scenario}_{i:02d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save metadata
            data['image_path'] = f"{scenario}_{i:02d}.png"
            dataset.append(data)
    
    # Save dataset metadata
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return dataset

if __name__ == "__main__":
    # Generate datasets
    print("Generating FBD-10 dataset...")
    fbd_dataset = generate_fbd_dataset()
    print(f"Generated {len(fbd_dataset)} FBD samples")
    
    print("Generating Circuit-10 dataset...")
    circuit_dataset = generate_circuit_dataset()
    print(f"Generated {len(circuit_dataset)} Circuit samples")
