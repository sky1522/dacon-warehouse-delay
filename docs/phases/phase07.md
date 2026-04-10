# Phase 7: Sample Weights + Diversity Ensemble

## Hypothesis
Upweighting extreme values (q90/q95/q99) and late timesteps will help models focus on hard cases.

## Implementation
- Sample weights: +0.15 (q90), +0.30 (q95), +0.60 (q99), +0.08 (time fraction)
- 4 diverse models: LGB raw+MAE, LGB log1p+Huber, XGB raw+MAE, Cat log1p+MAE
- 51 new time-series features: onset(8), expanding mean(30), nonlinear thresholds(7), time phase(6)

## Results
- CV MAE: [unknown]

## Lessons
- Sample weights provide modest improvement
- Model diversity (different transforms + losses) is key for ensemble gains
