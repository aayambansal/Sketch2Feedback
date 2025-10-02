# Paper Finalization Checklist for AI4EDU-2026

## Pre-Submission Checklist

### 0. Quick Wins (COMPLETED ✅)

- [x] **A. Resolved internal inconsistencies**: Marked pilot results clearly; removed conflicting claims
- [x] **B. Plan for N expansion**: Infrastructure ready for 200 samples per dataset
- [x] **C. Double-blind hygiene**: Anonymized; removed acknowledgments; removed GitHub URL
- [x] **D. Education literature grounding**: Added Hattie & Timperley [1]; Shute [11]; rubric design
- [x] **E. Prior diagram work positioning**: Added Mechanix [6]; circuit datasets [7-9]; Tangram [10]

### 1. Content Requirements

#### Abstract
- [x] Clear statement of problem (diagram feedback under-evaluated)
- [x] Contributions listed (benchmarks, pipeline, evaluation)
- [x] Key findings statement (changed to "hypothesize and design experiments")
- [x] Anonymized artifact release statement
- [ ] **TODO**: Update with final results when available

#### Introduction (Section 1)
- [x] Motivation: Why diagrams matter in STEM education
- [x] Gap: LMMs under-evaluated on student diagrams
- [x] Education-first framing with feedback theory [1]
- [x] Four contributions clearly listed
- [x] No premature claims (appropriate hedging)

#### Related Work (Section 2)
- [x] LMMs & visual instruction tuning [2,3]
- [x] Efficient VLMs [4,5]
- [x] FBD systems (Mechanix) [6]
- [x] Circuit recognition datasets [7-9]
- [x] Visual math benchmarks (Tangram) [10]
- [x] Feedback theory [1,11]
- [x] LLM-as-judge considerations [12]

#### Tasks & Datasets (Section 3)
- [x] Clear scenario descriptions (10 FBD + 10 Circuit)
- [x] Unified error taxonomy (8 types)
- [x] Generation approach (synthetic-first, real-lite)
- [x] Target sample size (≥200 per dataset)
- [x] Rubric design with education theory backing

#### Methods (Section 4)
- [x] Grammar-in-the-loop pipeline with 4 stages
- [x] Pseudocode provided
- [x] Baseline models clearly described (E2E-LMM, Vision-only)
- [x] Ablation designs listed
- [x] Efficiency considerations (small VLMs)

#### Evaluation (Section 5)
- [x] Data splits specified (200 train, 200 test)
- [x] Metrics defined (detection, localization, feedback, hallucination)
- [x] Rating protocol (human + LLM-as-judge)
- [x] Robustness experiments described
- [x] Statistical analysis plan (bootstrap 95% CI, corrections)

#### Results (Section 6)
- [x] Pilot results (n=5) clearly marked
- [x] Result tables with proper formatting
- [x] Honest reporting of limitations
- [ ] **TODO**: Update with full results (n=200) when ready
- [ ] **TODO**: Add bootstrap 95% CIs to all metrics
- [ ] **TODO**: Add statistical significance markers

#### Discussion (Section 7)
- [x] Key insights from pilot
- [x] Limitations clearly stated (synthetic, small pilot, simplified errors)
- [x] Validity & generalization discussion
- [x] Future work directions

#### Ethics (Section 8)
- [x] Privacy considerations (no student PII)
- [x] Validity threats acknowledged
- [x] LLM-as-judge cautions
- [x] Deployment considerations
- [x] Accessibility benefits

#### Conclusion (Section 9)
- [x] Summary of contributions
- [x] Key findings (hedged appropriately)
- [x] Future directions (5 clear items)

#### References
- [x] All 12 core references included
- [x] Education theory properly cited [1,11]
- [x] LMM work cited [2-5]
- [x] Diagram systems cited [6-10]
- [x] Judge reliability cited [12]
- [ ] **TODO**: Add any additional references for full paper

#### Appendices
- [x] Appendix A: Detailed rubric with examples (FBD + Circuit)
- [x] Appendix B: Prompts (VLM feedback + LLM-as-judge)
- [x] Appendix C: Dataset samples with annotations
- [x] Appendix D: Implementation details

### 2. Formatting Requirements

#### AAAI Workshop Format
- [ ] **TODO**: Convert from Markdown to AAAI LaTeX template
- [ ] **TODO**: Use `\documentclass[letterpaper]{article}` with AAAI style
- [ ] **TODO**: Double-column format (AAAI two-column style)
- [ ] **TODO**: Page limit: 5-7 pages (currently ~5,200 words ≈ 6-7 pages)
- [ ] **TODO**: Anonymous submission (no author names/affiliations)

#### Figures & Tables
- [ ] **TODO**: Create Figure 1 (pipeline overview diagram)
- [ ] **TODO**: Create Figure 2 (error taxonomy examples with before/after)
- [x] Table 1: FBD results with proper formatting
- [x] Table 2: Circuit results with proper formatting
- [ ] **TODO**: Add Table 3: Statistical comparisons with CIs
- [ ] **TODO**: Create Figure 3: Results visualization (bar charts with error bars)

#### Citations
- [x] All citations formatted consistently
- [x] No broken or incomplete citations
- [ ] **TODO**: Verify all URLs are accessible
- [ ] **TODO**: Add DOIs where available

### 3. Double-Blind Requirements

#### Anonymization
- [x] Author names removed
- [x] Affiliations removed
- [x] "Anonymous Authors, Paper #XXX" in header
- [x] GitHub URL removed (replaced with "anonymized repository")
- [x] Acknowledgments removed
- [ ] **TODO**: Remove any identifying information from code/data paths
- [ ] **TODO**: Scrub LaTeX metadata

#### Self-Citations
- [x] No self-citations in current draft
- [ ] **TODO**: If adding self-citations, use third-person ("Smith et al. [X]")

### 4. Content Quality

#### Writing Quality
- [x] Clear, concise writing
- [x] No jargon without explanation
- [x] Appropriate hedging (no overclaiming)
- [x] Logical flow between sections
- [ ] **TODO**: Final proofread for typos
- [ ] **TODO**: Check for passive voice overuse
- [ ] **TODO**: Verify all acronyms defined on first use

#### Technical Accuracy
- [x] No conflicting numbers
- [x] Pilot results clearly marked
- [x] Limitations acknowledged
- [x] Methods reproducible from description
- [ ] **TODO**: Verify all technical claims are supported

#### Pedagogical Grounding
- [x] Feedback theory properly cited [1,11]
- [x] Rubric linked to education research
- [x] Educational goals clearly stated
- [x] Practical deployment considerations discussed

### 5. Submission Logistics

#### AI4EDU-2026 Workshop Details
- **Workshop**: AI for Education (AI4EDU) @ AAAI-2026
- **Abstract deadline**: October 15, 2025
- **Full paper deadline**: October 22, 2025
- **Notification**: November 13, 2025
- **Submission portal**: OpenReview (workshop-specific track)
- **Format**: AAAI proceedings style, 5-7 pages
- **Double-blind**: Yes
- **Non-archival**: Yes (dual submission allowed)

#### Pre-Submission Tasks
- [ ] **TODO**: Convert to AAAI LaTeX template
- [ ] **TODO**: Generate PDF and check page count
- [ ] **TODO**: Verify all figures render correctly
- [ ] **TODO**: Create supplementary materials (code, data samples)
- [ ] **TODO**: Prepare anonymized code repository (GitHub/anonymous repo)
- [ ] **TODO**: Write 1-paragraph summary for OpenReview submission
- [ ] **TODO**: Select 3-5 keywords for submission
- [ ] **TODO**: Check PDF for anonymization (no metadata leaks)

#### OpenReview Submission
- [ ] **TODO**: Create OpenReview account (if needed)
- [ ] **TODO**: Upload PDF
- [ ] **TODO**: Upload supplementary materials (optional)
- [ ] **TODO**: Fill out submission form (title, abstract, keywords)
- [ ] **TODO**: Declare conflicts of interest
- [ ] **TODO**: Confirm double-blind compliance
- [ ] **TODO**: Submit before deadline (with buffer time!)

### 6. Post-Submission Tasks

#### Camera-Ready (After Acceptance)
- [ ] De-anonymize (add author names, affiliations)
- [ ] Add acknowledgments
- [ ] Add public GitHub URL
- [ ] Update with reviewer feedback
- [ ] Final proofread
- [ ] Submit camera-ready by deadline

#### Artifact Release
- [ ] Create public GitHub repository
- [ ] Add comprehensive README
- [ ] Include dataset generation code
- [ ] Include evaluation code
- [ ] Add sample data (10-20 examples)
- [ ] Add LICENSE (MIT or Apache 2.0)
- [ ] Add CITATION.bib
- [ ] Create DOI via Zenodo (optional)

### 7. Priority Actions for Next 48 Hours

#### Critical Path (Before Abstract Deadline - Oct 15)
1. [ ] **Scale dataset to 200 samples per dataset** (4-6 hours)
2. [ ] **Implement end-to-end LMM baseline** (4-6 hours)
3. [ ] **Run full experiments** (8-10 hours compute time)
4. [ ] **Update results tables with CIs** (2-3 hours)
5. [ ] **Convert to AAAI LaTeX format** (2-3 hours)
6. [ ] **Create Figures 1-3** (3-4 hours)
7. [ ] **Final proofread** (1-2 hours)
8. [ ] **Submit abstract by Oct 15** ⚠️ DEADLINE

#### Before Full Paper Deadline (Oct 22)
9. [ ] **Complete all ablation experiments** (4-6 hours)
10. [ ] **Collect teacher validation sketches** (2-3 hours)
11. [ ] **Run statistical comparisons** (2-3 hours)
12. [ ] **Update discussion with final results** (2-3 hours)
13. [ ] **Verify all checklist items** (2-3 hours)
14. [ ] **Submit full paper by Oct 22** ⚠️ DEADLINE

### 8. Common Rejection Reasons to Avoid

- [x] **Insufficient evaluation**: Addressed with 200-sample plan + bootstrap CIs
- [x] **Weak baselines**: Added end-to-end LMM baseline + ablations
- [x] **No statistical testing**: Planned bootstrap CIs + multiple comparison correction
- [x] **Poor motivation**: Grounded in education theory [1,11]
- [x] **Missing related work**: Comprehensive coverage of LMMs, diagrams, feedback
- [x] **Overclaiming**: Appropriate hedging throughout
- [x] **Reproducibility concerns**: Detailed methods + artifact release plan
- [x] **Synthetic-only evaluation**: Teacher validation set planned
- [ ] **Incomplete results**: Will address with full experiments

### 9. Reviewer Expectations (AI4EDU Workshop)

#### Technical Rigor
- [x] Clear problem definition
- [x] Appropriate baselines
- [ ] Statistical validation (in progress)
- [x] Ablation studies designed

#### Educational Relevance
- [x] Grounded in education research
- [x] Practical deployment considerations
- [x] Teacher involvement (validation set)
- [x] Feedback quality emphasis

#### Novelty
- [x] New benchmarks (FBD-10, Circuit-10)
- [x] Grammar-in-the-loop approach
- [x] Rubric-aligned evaluation
- [x] Hallucination measurement

#### Reproducibility
- [x] Detailed methods
- [x] Pseudocode provided
- [x] Implementation details in appendix
- [x] Artifact release planned

### 10. Final Sanity Checks

#### Before Abstract Submission (Oct 15)
- [ ] Abstract accurately reflects paper content
- [ ] Contributions are clear and novel
- [ ] No overclaiming (appropriate hedging)
- [ ] Education theory properly cited
- [ ] PDF renders correctly (no weird formatting)

#### Before Full Paper Submission (Oct 22)
- [ ] All sections complete (no TODOs left)
- [ ] Results tables filled with real data + CIs
- [ ] All figures created and referenced
- [ ] All references complete and formatted
- [ ] Page count within limits (5-7 pages)
- [ ] Double-blind compliance verified
- [ ] Supplementary materials prepared
- [ ] PDF metadata scrubbed (no author info)

---

## Quick Reference: Key Dates

| Date | Milestone |
|------|-----------|
| **Oct 15, 2025** | Abstract deadline ⚠️ |
| **Oct 22, 2025** | Full paper deadline ⚠️ |
| **Nov 13, 2025** | Notification |
| **TBD** | Camera-ready deadline |
| **TBD** | Workshop @ AAAI-2026 |

## Quick Reference: Target Metrics

| Dataset | Model | Target F1 | Target Actionability | Target Halluc. Rate |
|---------|-------|-----------|---------------------|-------------------|
| FBD-10 | Grammar+VLM | > 0.50 | > 4.0 | < 0.20 |
| FBD-10 | E2E-LMM | > 0.40 | > 3.0 | < 0.30 |
| Circuit-10 | Grammar+VLM | > 0.40 | > 3.5 | < 0.25 |
| Circuit-10 | E2E-LMM | > 0.30 | > 3.0 | < 0.35 |

*Note: These are aspirational targets; actual results may vary. Report honestly regardless of outcome.*

---

**Last Updated**: 2025-10-02  
**Status**: Paper draft revised; ready for full experiments  
**Next Action**: Scale datasets and run full evaluation suite
