# Phase 13 Step 5A: Bin9 Classifier Features (FAILED)

## Hypothesis
If we can identify bin 9 samples (target > 100) with a classifier, we can feed that information as features to the regression model.

## Implementation
- 7 new features:
  - 5 rule-based binary features (domain thresholds)
  - 1 OOF classifier probability (LGBM AUC ~0.92)
  - 1 layout-level bin9 rate (target encoding)
- Phase 13s1 structure (8 models)

## Results
- CV MAE: [unknown] (worsened vs baseline)

## Lessons (CRITICAL)
- **layout_bin9_rate bug**: With StratifiedGroupKFold(layout_id), validation layouts never appear in training fold
  - All validation rows get the global mean as their layout_bin9_rate
  - The feature becomes constant in validation — zero information
  - This is the same bug pattern that Phase 17's layout_residual_bias would have had
- **Classifier probability as feature**: Converting classification output to regression input doesn't help
  - The regression model already has access to all the features the classifier used
  - Adding P(extreme) is redundant — the model can learn its own implicit version
- **Key takeaway**: Any target-derived feature on layout_id is useless under layout-based GroupKFold
