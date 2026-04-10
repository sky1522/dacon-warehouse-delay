# Phase 1: Time Series Features

## Hypothesis
Lag, rolling, diff, cumsum features on 8 key dynamic columns will capture temporal patterns within scenarios.

## Implementation
- 24 lag features (8 cols x 3 lags)
- 32 rolling features (8 cols x 2 windows x 2 stats)
- 4 diff features, 4 cumsum features
- Total: 64 new features, 186 total

## Results
- CV MAE: 8.8508
- Public MAE: 10.249 (14th/249)

## Lessons
- Time series features gave a significant boost (~0.33 improvement from baseline 9.18)
- Established scenario-based grouping as the fundamental unit
