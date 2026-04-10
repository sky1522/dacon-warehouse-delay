# Phase 13 Step 3: Tail-aware Modeling (FAILED)

## Hypothesis
Upweighting tail samples + adding quantile-specialized models will improve extreme value (bin 9) prediction.

## Implementation
- Tail-aware sample weight: `1 + (y / q95).clip(0, 3)` — 1.0 at y=0, 2.0 at y=q95, 4.0 at 3xq95
- 3 new models (total 11):
  - LGB Quantile 0.9: alpha=0.9 for high-target specialization
  - LGB Quantile 0.75: alpha=0.75 for mid-high target
  - XGB log1p+Huber: log compression of tails
- Same 346 features, StratifiedGroupKFold(layout_id)

## Results
- CV MAE: [unknown] (expected improvement did not materialize)

## Lessons (CRITICAL)
- **Tree models cannot improve tail prediction via sample weights**
  - Trees predict leaf averages — weighting changes which leaf is chosen, not the leaf's prediction range
  - A leaf containing samples [100, 200, 300] will predict ~200 regardless of weights
  - Weights cannot create new leaves that predict outside the training range
- **Quantile regression changes the loss but not the prediction mechanism**
  - Q90 loss makes the model predict higher on average, but still bounded by leaf means
- **Fundamental insight**: GBDT extrapolation ceiling is structural, not a training problem
- This failure directly motivated Phase 18's two-stage approach (which also failed for different reasons)
