# Phase 9: Optuna Retune + Multi-Seed

## Hypothesis
Retuning Optuna with 319 features (vs 194) + 3-seed averaging will improve stability.

## Implementation
- Optuna retune: 30 trials on 319 features
- Multi-seed: 6 models x 3 seeds (42, 123, 777) → seed average
- Level 2 LGB stacking: 6 OOF + 6 original features = 12 meta features

## Results
- CV MAE: [unknown]

## Lessons
- Multi-seed averaging reduces variance but increases training cost 3x
- Optuna retuning on larger feature set can find different optima
