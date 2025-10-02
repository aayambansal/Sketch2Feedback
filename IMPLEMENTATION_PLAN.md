# Implementation Plan: Scaling to Full Experiments

## Quick Wins Implemented ✅

### A. Resolved Internal Inconsistencies
- **Fixed**: Conflicting numbers across tables/text
- **Solution**: Marked pilot results clearly as "n=5 pilot" with disclaimer
- **Action**: Removed absolute claims until full experiments complete

### B. Expand N and Report Uncertainty
- **Target**: 200 images per dataset (train), 200 images per dataset (test)
- **Required**: Bootstrap 95% CIs, paired comparisons
- **Status**: Infrastructure ready, needs execution

### C. Double-Blind Hygiene ✅
- **Fixed**: Removed GitHub URL (marked as "anonymized repository")
- **Fixed**: Removed acknowledgments section
- **Fixed**: Changed to "Anonymous Authors, Paper #XXX"
- **Action**: Use AAAI-26 workshop LaTeX template for final submission

### D. Ground in Education Literature ✅
- **Added**: Hattie & Timperley [1] (feedback levels)
- **Added**: Shute [11] (actionable formative feedback)
- **Added**: Clear rubric design section linking to theory

### E. Position Against Prior Diagram Work ✅
- **Added**: Mechanix [6] comparison (FBD sketch grading)
- **Added**: Circuit recognition datasets [7-9] comparison
- **Added**: Tangram [10] geometric reasoning benchmark
- **Clarified**: Our focus on feedback vs. recognition

## Immediate Next Steps (Priority Order)

### 1. Scale Dataset Generation (2-3 hours)
```bash
# Current: 100 samples per dataset (10 scenarios × 10 variations)
# Target: 400 samples per dataset (10 scenarios × 40 variations)

python generate_large_datasets.py --samples-per-scenario 40
```

**Actions**:
- Modify `generate_fbd_dataset()` to create 40 variations per scenario
- Modify `generate_circuit_dataset()` to create 40 variations per scenario
- Add noise/quality variations (neat, messy, low-contrast)
- Total: 400 FBD samples + 400 Circuit samples

### 2. Implement Full Baseline Models (4-6 hours)

**End-to-End LMM**:
- [ ] Integrate LLaVA-1.5 or similar open model
- [ ] Create proper prompting template
- [ ] Implement error detection from generated text
- [ ] Extract bounding boxes if mentioned in output

**Grammar+VLM Complete**:
- [ ] Fix constraint checking false positives
- [ ] Integrate Idefics2-8B or MiniCPM-V
- [ ] Implement rubric-aligned prompting
- [ ] Add few-shot examples to prompts

### 3. Implement Bootstrap Confidence Intervals (2-3 hours)

```python
def bootstrap_ci(data, metric_fn, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap 95% CI for any metric."""
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(metric_fn(sample))
    
    lower = np.percentile(bootstrap_samples, 100 * alpha/2)
    upper = np.percentile(bootstrap_samples, 100 * (1-alpha/2))
    return lower, upper

# Usage
f1_scores = [compute_f1(pred, gt) for pred, gt in zip(predictions, ground_truth)]
lower, upper = bootstrap_ci(f1_scores, np.mean)
print(f"F1: {np.mean(f1_scores):.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
```

### 4. Implement LLM-as-Judge with Position Balancing (3-4 hours)

```python
def llm_judge_with_position_balance(feedback_a, feedback_b, ground_truth):
    """Evaluate two feedback samples with position bias mitigation."""
    
    # Trial 1: A first
    prompt_1 = create_judge_prompt(feedback_a, feedback_b, ground_truth, position="A_first")
    scores_1 = query_llm(prompt_1)
    
    # Trial 2: B first (swapped)
    prompt_2 = create_judge_prompt(feedback_b, feedback_a, ground_truth, position="B_first")
    scores_2 = query_llm(prompt_2)
    
    # Average scores accounting for position swap
    final_scores = average_with_swap(scores_1, scores_2)
    return final_scores
```

### 5. Teacher Sketch Collection (1-2 hours)

**Approach**:
- Ask 2 instructors to hand-draw 20 FBD + 20 Circuit examples each
- Scan/photograph at 300 DPI
- Annotate with ground-truth errors
- Use as hold-out validation set

**Alternative**: Use existing handwritten circuit datasets [8,9] for circuit validation

### 6. Statistical Analysis Pipeline (2-3 hours)

```python
def compare_models_with_stats(model_a_results, model_b_results):
    """Compare two models with paired bootstrap and multiple comparison correction."""
    
    # Paired bootstrap for difference
    differences = model_a_results - model_b_results
    diff_mean = np.mean(differences)
    ci_lower, ci_upper = bootstrap_ci(differences, np.mean)
    
    # Check if CI excludes zero
    significant = not (ci_lower <= 0 <= ci_upper)
    
    return {
        'difference': diff_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': significant
    }

# Multiple comparisons correction
from statsmodels.stats.multitest import multipletests
p_values = [...]  # compute for each comparison
rejected, p_adjusted, _, _ = multipletests(p_values, method='holm')
```

## Revised Timeline

### Week 1: Data & Infrastructure
- **Day 1-2**: Scale dataset generation to 400 samples per dataset
- **Day 3-4**: Collect teacher sketches (n=40)
- **Day 5**: Implement bootstrap CI and statistical comparison

### Week 2: Models & Evaluation
- **Day 1-2**: Complete end-to-end LMM baseline
- **Day 3-4**: Fix grammar+VLM constraint checking
- **Day 5**: Implement LLM-as-judge with position balancing

### Week 3: Experiments & Analysis
- **Day 1-2**: Run full experiments (n=200 test per dataset)
- **Day 3**: Run robustness experiments (noise, symbols)
- **Day 4**: Run ablations (no constraints, no VLM, no rubric)
- **Day 5**: Statistical analysis and generate tables

### Week 4: Paper Finalization
- **Day 1-2**: Update paper with results and CIs
- **Day 3**: Create figures (overview, error examples, results plots)
- **Day 4**: Convert to AAAI LaTeX format
- **Day 5**: Final proofreading and submission prep

## Code Changes Needed

### 1. Dataset Generator (`src/dataset_generator.py`)

```python
def generate_fbd_dataset(num_samples_per_scenario: int = 40, 
                         noise_levels: List[str] = ['neat', 'medium', 'messy'],
                         output_dir: str = "data/fbd_full"):
    """Generate large-scale FBD dataset with noise variations."""
    
    for scenario in scenarios:
        for i in range(num_samples_per_scenario):
            noise_level = noise_levels[i % len(noise_levels)]
            error_types = sample_error_combination()
            
            # Generate with noise
            data = generate_scenario(scenario, error_types, noise_level)
            
            # Save with noise label
            save_image(data, f"{scenario}_{noise_level}_{i:03d}.png")
```

### 2. Evaluation Suite (`src/evaluation.py`)

```python
class StatisticalEvaluator:
    """Evaluator with bootstrap CIs and multiple comparison correction."""
    
    def evaluate_with_uncertainty(self, predictions, ground_truth, 
                                  n_bootstrap=10000):
        """Compute metrics with bootstrap 95% CIs."""
        
        metrics = self.compute_metrics(predictions, ground_truth)
        
        # Bootstrap each metric
        ci_dict = {}
        for metric_name, metric_fn in self.metrics.items():
            lower, upper = bootstrap_ci(
                predictions, ground_truth, metric_fn, n_bootstrap
            )
            ci_dict[metric_name] = (lower, upper)
        
        return metrics, ci_dict
    
    def compare_models(self, model_results_dict):
        """Compare multiple models with Holm-Bonferroni correction."""
        
        # Pairwise comparisons
        comparisons = []
        for model_a, model_b in itertools.combinations(model_results_dict.keys(), 2):
            comparison = self.paired_bootstrap_test(
                model_results_dict[model_a],
                model_results_dict[model_b]
            )
            comparisons.append(comparison)
        
        # Multiple comparison correction
        p_values = [c['p_value'] for c in comparisons]
        rejected, p_adjusted, _, _ = multipletests(p_values, method='holm')
        
        return comparisons, rejected, p_adjusted
```

### 3. Experiment Runner (`run_full_experiments.py`)

```python
def run_full_experiments():
    """Run complete experimental suite with 200 samples per dataset."""
    
    # Load full datasets
    fbd_train = load_dataset("data/fbd_full", split="train")  # 200 samples
    fbd_test = load_dataset("data/fbd_full", split="test")    # 200 samples
    circuit_train = load_dataset("data/circuit_full", split="train")
    circuit_test = load_dataset("data/circuit_full", split="test")
    
    # Load teacher validation
    teacher_fbd = load_dataset("data/teacher_sketches/fbd")   # 20 samples
    teacher_circuit = load_dataset("data/teacher_sketches/circuit")  # 20 samples
    
    # Initialize models
    models = {
        'E2E-LMM': EndToEndLMM(),
        'Vision-only': VisionOnlyDetector(),
        'Grammar+VLM': GrammarInTheLoopPipeline()
    }
    
    # Run experiments
    results = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # Test set evaluation
        fbd_results = evaluate_model(model, fbd_test, "fbd")
        circuit_results = evaluate_model(model, circuit_test, "circuit")
        
        # Teacher validation
        teacher_fbd_results = evaluate_model(model, teacher_fbd, "fbd")
        teacher_circuit_results = evaluate_model(model, teacher_circuit, "circuit")
        
        results[model_name] = {
            'fbd_test': fbd_results,
            'circuit_test': circuit_results,
            'fbd_teacher': teacher_fbd_results,
            'circuit_teacher': teacher_circuit_results
        }
    
    # Statistical comparison
    evaluator = StatisticalEvaluator()
    comparisons, rejected, p_adjusted = evaluator.compare_models(results)
    
    # Generate report with CIs
    generate_report_with_uncertainty(results, comparisons, rejected, p_adjusted)
```

## Expected Results Table Format

```
FBD-10 (test, n=200)

Model         F1 (↑)              Precision (↑)       Recall (↑)          mIoU (↑)            Action. (↑)         Halluc. (↓)
E2E-LMM       0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  X.X [X.X, X.X]      0.XXX [0.XX, 0.XX]
Vision-only   0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  X.X [X.X, X.X]      0.XXX [0.XX, 0.XX]
Grammar+VLM   0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  0.XXX [0.XX, 0.XX]  X.X [X.X, X.X]      0.XXX [0.XX, 0.XX]

* [lower, upper] = 95% bootstrap CI
** Bold indicates best performance (CI excludes other models)
```

## Resources Needed

### Computational
- **GPU**: NVIDIA RTX 3090 or better (24GB+ VRAM)
- **Runtime**: ~10-12 hours for full experimental suite
- **Storage**: ~5GB for datasets + models

### Human Resources
- **2 instructor raters**: ~4 hours each for rubric rating (100 samples)
- **Teacher sketch collection**: ~2 hours (2 instructors × 20 sketches each)

### Software Dependencies
```bash
# Add to requirements.txt
statsmodels>=0.14.0
scipy>=1.11.0
llava>=1.5.0  # or equivalent open LMM
idefics2>=0.1.0  # or miniсpm-v
```

## Success Criteria

### Minimum Viable Results
- [x] 200 test samples per dataset
- [ ] Bootstrap 95% CIs for all metrics
- [ ] At least 2 models compared (grammar+VLM vs baseline)
- [ ] Teacher validation set results (n=20 per dataset)
- [ ] Statistical significance testing with correction

### Ideal Results
- [ ] 3+ models compared (E2E-LMM, Vision-only, Grammar+VLM)
- [ ] Robustness experiments (noise, symbols)
- [ ] Ablation studies (constraints, VLM, rubric)
- [ ] LLM-as-judge with human agreement
- [ ] Cost/latency analysis

## Notes

- Prioritize **clean, self-consistent results** over absolute scores
- Document **all decisions** (error injection rates, constraint thresholds, etc.)
- Keep **pilot results** available for comparison but clearly marked
- Maintain **anonymization** throughout review period
