# Phase 16: 2nd-order Feature Engineering (CURRENT BEST)

## Hypothesis
2nd-order features (lag of engineered features, rolling of rolling, cumulative stats, cross-feature interactions) will capture higher-order patterns.

## Implementation
- A) Lag features: 14 cols x 3 lags = 42
- B) Rolling features: 10 cols x 2 windows x 2 stats = 40
- C) Cumulative features: 7 cols x 4 stats = 28
- D) Interaction features: 15
- E) Lag-based scenario aggregation: 7 cols x 3 stats = 21
- Selection: top 150 from ~146 new features
- Final: ~696 features (546 + 150)
- 7 models: LGB x3 + XGB + Cat x2 + MLP (TabNet failed)

## Results
- CV MAE: 8.4403
- Public MAE: 9.87947 (5th place)

## Bug Fixes
- MLP scaler leakage: global StandardScaler → fold-internal fit/transform
- Checkpoint metadata validation: pipeline_version + feature_cols mismatch → cache invalidation
- LGB selector deterministic: deterministic=True, force_col_wise=True, n_jobs=1

## Lessons
- 2nd-order features provide meaningful signal beyond 1st-order
- MLP scaler leakage was silently hurting performance — always use fold-isolated scaling
- This established the checkpoint validation pattern used in all subsequent phases
