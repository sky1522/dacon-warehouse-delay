# Phase 3A/3B: Feature Selection + Optuna

## Hypothesis
- 3A: Remove low-importance features, add layout-derived features, strengthen regularization
- 3B: Residual-based features + Optuna hyperparameter tuning

## Implementation
### Phase 3A
- Feature selection (remove bottom 30%)
- Layout features: robot_per_packstation, charger_density, etc.
- Layout GroupKFold for unseen layout simulation

### Phase 3B
- 8 residual-based features (orders_per_packstation, demand_density, etc.)
- Dual model strategy (base + high-delay specialist)
- Optuna 50 trials for LightGBM

## Results
- CV MAE: [unknown]

## Lessons
- Established the iterative feature engineering + Optuna tuning pattern
- Residual analysis became a standard diagnostic tool
