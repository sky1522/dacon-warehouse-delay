# Phase 10: Neural Networks (MLP + TabNet)

## Hypothesis
Adding Keras MLP and TabNet to tree ensemble increases model diversity and reduces CV-Public gap.

## Implementation
- 6 tree models + Keras MLP (512-256-128-64, log1p) + TabNet (n_d=32, entmax)
- 8-model stacking

## Results
- CV MAE: [unknown]

## Lessons
- MLP adds genuine diversity to tree-only ensemble
- TabNet is unstable (fails intermittently) — always wrap in try/except
- MLP needs fold-isolated StandardScaler (discovered later in Phase 16)
