# Phase 13 Step 4: Bin 9 Extreme Target EDA

## Hypothesis
Bin 9 (target > 100) samples have identifiable patterns that can be exploited for better prediction.

## Implementation
- KS test: bin9 vs non-bin9 for all features
- KMeans clustering within bin9 (sub-patterns)
- 2-feature combination analysis for bin9 occurrence rate
- LGBMClassifier AUC for bin9 detection
- Test set bin9 probability distribution

## Results
- Strong separators (KS > 0.5): [unknown] features
- LGBM classifier AUC: ~0.92
- Bin9 is concentrated in hard layouts + high pack_utilization + high order_inflow

## Lessons
- Bin9 is classifiable (AUC 0.92) but converting classification → regression is hard
- Knowing "this is extreme" doesn't help predict "how extreme"
- → Led to Phase 13s5a (failed) and Phase 18 (also failed)
