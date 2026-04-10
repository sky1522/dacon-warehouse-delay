# Phase 11: Adversarial Validation + Transformer

## Hypothesis
- Adversarial weights will close CV-Public gap by upweighting test-like training samples
- Transformer on raw sequences will capture temporal dependencies trees miss

## Implementation
- Adversarial validation: LGB classifier train vs test → combined_weight
- Transformer: d_model=128, nhead=8, 3 layers, raw 90 features sequence input
- 9 models total (6 tree + MLP + TabNet + Transformer)

## Results
- CV MAE: [unknown]

## Lessons
- Adversarial validation had no effect (train/test distribution gap is small)
- Transformer on raw features underperformed — needs engineered features
- Both strategies abandoned in subsequent phases
