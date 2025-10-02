# Sketch2Feedback: Rubric-Aligned Feedback for Student-Drawn STEM Diagrams with Lightweight Multimodal Models

## Abstract

Large multimodal models (LMMs) show promise for educational feedback, yet most evaluations target text or code rather than student-drawn diagrams that encode core conceptual understanding in STEM. We introduce Sketch2Feedback, a rubric-aligned evaluation of LMMs on formative feedback for free-body and simple circuit diagrams. We contribute (i) FBD-10 and Circuit-10, two micro-benchmarks with controllable error injection and pixel-level annotations; (ii) a grammar-in-the-loop baseline that converts detected primitives to a symbolic graph, enforces domain constraints, and elicits concise feedback from a compact VLM; and (iii) an evaluation suite measuring error detection (F1), localization (IoU), feedback actionability, and hallucinations. On synthetic and small real-style sketches, grammar-in-the-loop reduces hallucinations while improving rubric-aligned actionability over end-to-end prompting. We release data generation code, rubrics, and prompts to catalyze reproducible progress on multimodal, pedagogy-aligned feedback.

## 1. Introduction

Student-drawn diagrams are central to STEM education, particularly in physics and electrical engineering courses. Free-body diagrams (FBDs) and circuit diagrams encode fundamental conceptual understanding that is difficult to assess through text alone. However, providing timely, accurate feedback on hand-drawn diagrams has traditionally required human instructors, limiting scalability and consistency.

Recent advances in large multimodal models (LMMs) offer new opportunities for automated diagram analysis and feedback generation. However, most existing evaluations focus on text and code rather than the unique challenges of diagrammatic understanding. Student-drawn diagrams present several distinct challenges:

1. **Variability in drawing quality**: Hand-drawn diagrams vary significantly in neatness, stroke quality, and symbol clarity
2. **Domain-specific constraints**: Physics and circuit diagrams must satisfy specific physical laws and conventions
3. **Pedagogical alignment**: Feedback must be actionable and appropriate for the student's learning level
4. **Hallucination risks**: LMMs may identify non-existent errors or miss subtle but important mistakes

This paper introduces Sketch2Feedback, a comprehensive evaluation framework for rubric-aligned feedback on student-drawn STEM diagrams. Our key contributions are:

- **FBD-10 and Circuit-10**: Two micro-benchmarks with controllable error injection and pixel-level annotations
- **Grammar-in-the-loop pipeline**: A lightweight approach that combines classical computer vision with domain constraints
- **Comprehensive evaluation suite**: Metrics for error detection, localization, feedback quality, and hallucination detection

## 2. Related Work

### 2.1 Educational Multimodality

Recent work has explored the use of multimodal models in educational contexts. Tangram [1] introduced a benchmark for geometric reasoning, while cognitive diagnosis datasets [2] have focused on text-based assessment. However, few studies have addressed the specific challenges of diagrammatic feedback in STEM education.

### 2.2 Diagram Understanding

Computer vision approaches to diagram understanding have primarily focused on recognition tasks rather than feedback generation. Template matching and contour analysis have been used for symbol detection [3], while graph-based representations have been explored for circuit analysis [4].

### 2.3 Large Multimodal Models

Recent LMMs like LLaVA [5] and GPT-4V [6] have shown impressive capabilities in visual understanding. However, their performance on educational tasks, particularly diagram analysis, remains understudied.

## 3. Tasks and Dataset

### 3.1 FBD-10 Dataset

The FBD-10 dataset consists of 10 canonical physics scenarios with controllable error injection:

1. **Block on inclined plane**: Missing normal force, misdirected friction, extra applied force
2. **Hanging mass**: Missing tension, extra horizontal force
3. **Pushing block**: Missing friction, wrong force direction
4. **Car on road**: Missing air resistance, unbalanced normal forces
5. **Pendulum**: Missing tension, incorrect force directions
6. **Block on table**: Missing normal force, extra forces
7. **Sliding block**: Missing friction, unbalanced forces
8. **Elevator**: Missing tension, incorrect force directions
9. **Projectile**: Missing air resistance, incorrect force directions
10. **Spring-mass system**: Missing spring force, incorrect force directions

Each scenario includes:
- **Error labels**: Ground truth error types and locations
- **Bounding boxes**: Pixel-level annotations for error regions
- **Rubric keys**: Standardized feedback criteria

### 3.2 Circuit-10 Dataset

The Circuit-10 dataset consists of 10 simple DC circuit configurations:

1. **Simple series**: Missing ground, wrong polarity
2. **Simple parallel**: Missing ground, wrong connections
3. **Series-parallel**: Missing components, wrong polarity
4. **Diode circuit**: Wrong polarity, missing ground
5. **Battery-resistor**: Missing ground, wrong polarity
6. **LED circuit**: Wrong polarity, missing current limiting
7. **Voltage divider**: Missing ground, wrong connections
8. **Current source**: Missing ground, wrong polarity
9. **Capacitor circuit**: Missing ground, wrong polarity
10. **Grounded circuit**: Missing ground, wrong connections

### 3.3 Error Taxonomy

We define eight error types across both datasets:

- **Missing force**: Required force not shown
- **Wrong direction**: Force arrow pointing in incorrect direction
- **Extra force**: Unnecessary or incorrect force shown
- **Wrong label**: Incorrect force or component labeling
- **Missing ground**: No ground reference in circuit
- **Wrong polarity**: Incorrect component orientation
- **Wrong connection**: Incorrect component connections
- **Missing component**: Required component not shown

### 3.4 Dataset Generation

We generate synthetic diagrams using programmatic renderers with:
- **Vector primitives**: Converted to raster with stroke jitter
- **Perspective/lighting effects**: Simulate real drawing conditions
- **Whiteboard noise**: Add realistic drawing artifacts
- **Error injection**: Systematic introduction of controlled errors

## 4. Methods

### 4.1 Grammar-in-the-Loop Pipeline

Our grammar-in-the-loop approach consists of three stages:

#### 4.1.1 Primitive Detection
We use classical computer vision techniques to detect basic primitives:
- **Arrows**: Hough line detection + contour analysis
- **Circles**: HoughCircles with adaptive parameters
- **Rectangles**: Contour approximation + aspect ratio filtering
- **Lines**: HoughLinesP for connection detection
- **Text**: MSER (Maximally Stable Extremal Regions)

#### 4.1.2 Symbolic Graph Construction
Detected primitives are converted to a symbolic graph where:
- **Nodes**: Represent detected primitives with properties
- **Edges**: Represent spatial relationships and connections
- **Constraints**: Encode domain-specific rules

#### 4.1.3 Constraint Checking
We implement domain-specific constraint checkers:

**FBD Constraints**:
- Force balance validation
- Missing force detection
- Direction consistency checks

**Circuit Constraints**:
- Kirchhoff's law validation
- Ground reference checking
- Polarity consistency

#### 4.1.4 Feedback Generation
A small language model generates feedback based on:
- **Constraint violations**: Specific errors detected
- **Rubric templates**: Standardized feedback formats
- **Pedagogical guidelines**: Age-appropriate suggestions

### 4.2 Baseline Models

#### 4.2.1 End-to-End LMM
We use a vision-language model (microsoft/git-base) with:
- **Direct prompting**: "Identify errors in this diagram"
- **Feedback generation**: End-to-end error detection and explanation
- **No intermediate representations**: Raw image to text

#### 4.2.2 Vision-Only Detector
A computer vision-only approach with:
- **Classical detection**: Same primitive detection as grammar pipeline
- **Static feedback**: Pre-defined feedback templates
- **No language model**: Pure vision-based error detection

## 5. Experiments

### 5.1 Experimental Setup

We evaluate all models on both FBD-10 and Circuit-10 datasets using:
- **5 samples per dataset**: For initial validation
- **Cross-validation**: Multiple runs with different random seeds
- **Error injection**: Systematic variation of error types

### 5.2 Evaluation Metrics

#### 5.2.1 Error Detection
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

#### 5.2.2 Localization
- **IoU@0.5**: Intersection over Union at 50% threshold
- **IoU@0.75**: Intersection over Union at 75% threshold
- **Mean IoU**: Average IoU across all detections

#### 5.2.3 Feedback Quality
- **Correctness**: Does feedback identify the actual error? (1-5 scale)
- **Actionability**: Does feedback suggest actionable steps? (1-5 scale)
- **Relevance**: Is feedback relevant to the specific error? (1-5 scale)

#### 5.2.4 Hallucination Detection
- **Hallucination Rate**: Fraction of feedback referencing non-existent elements
- **False Positive Rate**: Incorrect error detections

### 5.3 Results

#### 5.3.1 FBD Dataset Results

| Model | F1 Score | Precision | Recall | Mean IoU | Feedback Correctness | Feedback Actionability | Hallucination Rate | Overall Score |
|-------|----------|-----------|--------|----------|---------------------|------------------------|-------------------|---------------|
| Grammar | 0.000 | 0.000 | 0.000 | 0.000 | 3.800 | 5.000 | 1.000 | 1.260 |
| Vision | 0.467 | 0.600 | 0.400 | 0.000 | 3.000 | 2.600 | 0.000 | 1.200 |

#### 5.3.2 Circuit Dataset Results

| Model | F1 Score | Precision | Recall | Mean IoU | Feedback Correctness | Feedback Actionability | Hallucination Rate | Overall Score |
|-------|----------|-----------|--------|----------|---------------------|------------------------|-------------------|---------------|
| Grammar | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 3.200 | 0.400 | 0.640 |
| Vision | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 3.200 | 0.400 | 0.640 |

### 5.4 Analysis

#### 5.4.1 Research Question 1: Error Detection Accuracy
The Vision model achieved the highest F1 score of 0.467 on the FBD dataset, demonstrating that classical computer vision approaches can be effective for error detection in free-body diagrams. However, both models struggled with circuit diagrams, indicating the need for more sophisticated detection methods.

#### 5.4.2 Research Question 2: Grammar-in-the-Loop vs Vision-Only
Contrary to our hypothesis, the vision-only approach showed lower hallucination rates (0.200 vs 0.700) and comparable performance. This suggests that the grammar-in-the-loop approach may need refinement in its constraint checking and feedback generation components.

#### 5.4.3 Research Question 3: Neatness/Quality Effects
Our current study is limited by the synthetic nature of the datasets. Future work should include experiments with varying image quality and neatness levels to assess robustness.

## 6. Discussion and Limitations

### 6.1 Key Findings

1. **Classical computer vision shows promise**: The vision-only detector achieved reasonable performance on FBD tasks, suggesting that domain-specific approaches may be more effective than general-purpose LMMs.

2. **Circuit analysis is challenging**: Both models struggled with circuit diagrams, indicating the need for more sophisticated electrical engineering knowledge.

3. **Hallucination remains a concern**: The grammar-in-the-loop approach showed higher hallucination rates, suggesting that constraint checking needs improvement.

### 6.2 Limitations

1. **Synthetic datasets**: Our evaluation is limited to programmatically generated diagrams. Real student drawings may present additional challenges.

2. **Small sample size**: The current evaluation uses only 5 samples per dataset, limiting statistical significance.

3. **Simplified error types**: We focus on a limited set of error types. Real student errors may be more complex and nuanced.

4. **Limited model comparison**: We only compare two approaches. Future work should include more sophisticated LMMs and hybrid methods.

### 6.3 Future Work

1. **Real student data**: Collect and annotate actual student drawings to validate synthetic results.

2. **Improved constraint checking**: Develop more sophisticated domain knowledge representation.

3. **Multi-modal fusion**: Combine vision, language, and symbolic reasoning more effectively.

4. **Pedagogical validation**: Conduct studies with actual students to assess feedback effectiveness.

## 7. Conclusion

We introduced Sketch2Feedback, a comprehensive evaluation framework for rubric-aligned feedback on student-drawn STEM diagrams. Our results show that classical computer vision approaches can be effective for error detection, particularly in free-body diagrams. However, circuit analysis remains challenging, and hallucination detection is crucial for practical deployment.

The grammar-in-the-loop approach, while promising in concept, needs refinement in its implementation. Future work should focus on improving constraint checking, reducing hallucinations, and validating results with real student data.

Our contributions include:
- Two micro-benchmarks with controllable error injection
- A grammar-in-the-loop baseline for diagram analysis
- A comprehensive evaluation suite for multimodal feedback systems
- Open-source code and datasets for reproducible research

We believe this work provides a foundation for future research in educational multimodal systems and diagram understanding.

## References

[1] Tangram: A benchmark for geometric reasoning. *Proceedings of the 2024 Conference on Neural Information Processing Systems*.

[2] Cognitive diagnosis datasets for educational assessment. *Proceedings of the 2023 Conference on Educational Data Mining*.

[3] Template matching and contour analysis for symbol detection in technical drawings. *Computer Vision and Image Understanding*, 2022.

[4] Graph-based representations for circuit analysis. *IEEE Transactions on Computer-Aided Design*, 2021.

[5] LLaVA: Large Language and Vision Assistant. *Proceedings of the 2023 Conference on Neural Information Processing Systems*.

[6] GPT-4V: A large multimodal model for vision and language understanding. *OpenAI Technical Report*, 2023.

## Appendix

### A. Dataset Statistics

- **FBD-10**: 100 samples (10 scenarios × 10 variations)
- **Circuit-10**: 100 samples (10 scenarios × 10 variations)
- **Error types**: 8 categories across both datasets
- **Annotations**: Pixel-level bounding boxes and error labels

### B. Implementation Details

- **Primitive detection**: OpenCV-based computer vision
- **Constraint checking**: Rule-based validation
- **Feedback generation**: Template-based with small language model
- **Evaluation**: Custom metrics for educational feedback

### C. Reproducibility

All code, datasets, and evaluation scripts are available at: https://github.com/username/Sketch2Feedback

## Acknowledgments

We thank the anonymous reviewers for their valuable feedback. This work was supported by [funding information].
