# Phase 8: Competitor Features + Stacking

## Hypothesis
54 competitor-inspired features + 2-level stacking will push beyond simple weighted averaging.

## Implementation
- 54 new features: robot state decomposition(6), demand/capacity ratios(15), complex interactions(12), layout density(8), missing(3), rolling deviation(10)
- 20 expanding extension features (std + max)
- Level 1: 6 models (LGB x3, XGB, Cat x2)
- Level 2: Ridge + LGB meta model (12 meta features)

## Results
- CV MAE: [unknown]

## Lessons
- Stacking adds complexity but marginal improvement over Nelder-Mead weighted averaging
- Ridge meta model is safer than LGB meta model (less overfitting risk)
