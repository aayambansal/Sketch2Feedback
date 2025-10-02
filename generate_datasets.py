"""
Generate synthetic datasets for Sketch2Feedback project.
"""

import os
import sys

# Add src to path
sys.path.append('src')

from dataset_generator import generate_fbd_dataset, generate_circuit_dataset

def main():
    """Generate both datasets."""
    print("Generating Sketch2Feedback datasets...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Generate FBD-10 dataset
    print("Generating FBD-10 dataset...")
    fbd_dataset = generate_fbd_dataset(num_samples_per_scenario=10, output_dir="data/fbd_10")
    print(f"Generated {len(fbd_dataset)} FBD samples")
    
    # Generate Circuit-10 dataset
    print("Generating Circuit-10 dataset...")
    circuit_dataset = generate_circuit_dataset(num_samples_per_scenario=10, output_dir="data/circuit_10")
    print(f"Generated {len(circuit_dataset)} Circuit samples")
    
    print("Dataset generation completed!")
    print("Datasets saved in 'data/' directory")

if __name__ == "__main__":
    main()
