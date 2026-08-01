# External Dataset Statistics Report — Synthetic Demonstration Examples

**Total ML training rows**: 2
**Unique documents**: 2
**Unique queries**: 2
**Average tokens per chunk**: 9.00

## Binary Label Distribution
- Irrelevant (0): 0 (0.0%)
- Relevant (1): 2 (100.0%)

## Relevance Score Distribution (0–3)
- Score 1: 1 (50.0%)
- Score 3: 1 (50.0%)

## Difficulty Level Distribution
- external: 2

## Dataset Compatibility & Integration Notes
- External datasets are mapped into `MLTrainingExample` dataclasses.
- CUAD maps clause annotations to binary 0/1 and graded relevance 0 or 3.
- LexGLUE compatible tasks (EURLEX, ECtHR, UNFAIR-ToS) map task labels to binary 0/1.
