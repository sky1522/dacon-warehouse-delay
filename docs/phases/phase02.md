# Phase 2: Multi-model Ensemble

## Hypothesis
Combining LightGBM, CatBoost, XGBoost will reduce variance through model diversity.

## Implementation
- LightGBM: CV 8.8508
- CatBoost (depth=6): CV 8.8815
- XGBoost (depth=6): CV 8.9126
- Optimal weights: LGB=0.52, Cat=0.33, XGB=0.15

## Results
- Ensemble CV: 8.8253
- Improvement from Phase 1: -0.0255

## Lessons
- Even simple weighted averaging helps
- Bug found: test row order mismatch (original_idx preservation needed)
