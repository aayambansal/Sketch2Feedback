# Sketch2Feedback: Rubric-Aligned Feedback for Student-Drawn STEM Diagrams

## Overview

Sketch2Feedback evaluates and improves rubric-aligned formative feedback on student-drawn STEM diagrams (e.g., free-body diagrams and simple circuits) using a lightweight "grammar-in-the-loop" pipeline that runs on commodity hardware.

## Research Questions

- **RQ1**: How accurately can LMMs detect pedagogically salient errors in student-style diagrams?
- **RQ2**: Does a grammar-in-the-loop approach reduce hallucinations and improve actionable feedback vs. end-to-end LMMs?
- **RQ3**: How do neatness/scan quality and symbol variation affect detection and feedback quality?

## Dataset

- **FBD-10**: 10 canonical physics scenarios with controllable errors (missing normal force, misdirected friction, extra force, mislabelled vectors, etc.)
- **Circuit-10**: Simple DC circuits (series/parallel; diode orientation; missing ground; polarity inconsistencies)

## Baselines

1. **End-to-end LMM**: Open vision-language model prompted for error finding + feedback
2. **Grammar-in-the-loop**: Classical detection + constraints + small VLM for feedback generation
3. **Vision-only detector**: Static canned feedback (ablates the LMM)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Generate synthetic datasets
python generate_datasets.py

# Run experiments
python run_experiments.py

# Evaluate results
python evaluate.py
```

## Results

See `results/` directory for experimental results and analysis.
