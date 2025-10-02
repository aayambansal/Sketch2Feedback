# Sketch2Feedback: Simplified Experimental Results

## Dataset Overview
- **FBD-10**: 5 samples from 10 canonical physics scenarios
- **Circuit-10**: 5 samples from 10 simple DC circuits

## Results Summary

### FBD Dataset

#### Grammar Model
- **F1 Score**: 0.000
- **Precision**: 0.000
- **Recall**: 0.000
- **Mean IoU**: 0.000
- **Feedback Correctness**: 3.800
- **Feedback Actionability**: 5.000
- **Hallucination Rate**: 1.000
- **Overall Score**: 1.260

#### Vision Model
- **F1 Score**: 0.467
- **Precision**: 0.600
- **Recall**: 0.400
- **Mean IoU**: 0.000
- **Feedback Correctness**: 3.000
- **Feedback Actionability**: 2.600
- **Hallucination Rate**: 0.000
- **Overall Score**: 1.200

### Circuit Dataset

#### Grammar Model
- **F1 Score**: 0.000
- **Precision**: 0.000
- **Recall**: 0.000
- **Mean IoU**: 0.000
- **Feedback Correctness**: 1.000
- **Feedback Actionability**: 3.200
- **Hallucination Rate**: 0.400
- **Overall Score**: 0.640

#### Vision Model
- **F1 Score**: 0.000
- **Precision**: 0.000
- **Recall**: 0.000
- **Mean IoU**: 0.000
- **Feedback Correctness**: 1.000
- **Feedback Actionability**: 3.200
- **Hallucination Rate**: 0.400
- **Overall Score**: 0.640

## Key Findings

### Best Overall Performance
- **Model**: Grammar on FBD dataset
- **Overall Score**: 1.260

### Best Error Detection
- **Model**: Vision on FBD dataset
- **F1 Score**: 0.467

### Lowest Hallucination Rate
- **Model**: Vision on FBD dataset
- **Hallucination Rate**: 0.000

## Research Question Answers

### RQ1: Error Detection Accuracy
The Vision model achieved the highest F1 score of 0.467 on the FBD dataset.

### RQ2: Grammar-in-the-Loop vs Vision-Only
- **Grammar-in-the-Loop F1**: 0.000
- **Vision-Only F1**: 0.233
- **Grammar-in-the-Loop Hallucination Rate**: 0.700
- **Vision-Only Hallucination Rate**: 0.200

**Result**: Vision-only approach shows lower hallucination rate.

### RQ3: Neatness/Quality Effects
Analysis of neatness and scan quality effects requires additional experiments with varying image quality. This is a limitation of the current study.

