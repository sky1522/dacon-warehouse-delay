# Phase 5: Kaggle Winner Strategies

## Hypothesis
GroupBy aggregation features (300+) and target lag features will capture cross-sample patterns.

## Implementation
- 300 GroupBy features (4 group keys x 15 agg targets x 5 stats)
- Target lag features (lag1-3, rolling mean, cummax, cummean)
- Model A: LGB without target lags
- Model B: LGB with target lags (recursive test prediction)
- CatBoost + 3-model ensemble

## Results
- CV MAE: [unknown]

## Lessons
- GroupBy features are powerful but need selection (many are redundant)
- Target lag features risk leakage on test (recursive prediction needed)
