# Phase 18: Two-Stage Hurdle Model (FAILED)

## Hypothesis
GBDT cannot extrapolate beyond training range (structural limitation). A two-stage model separating normal (GBDT) and extreme (MLP with linear output) prediction should break this ceiling.

## Implementation
- Stage 2A (Normal): Phase 16 GBDT ensemble reuse
- Stage 1 (Classifier): LGBMClassifier, threshold T search (50/80/100), class_weight='balanced'
- Stage 2B (Extreme MLP): 128-64-32 GELU, raw target MAE loss, linear output, fold-isolated scaler
  - Trained only on extreme samples (y > T)
  - Top 150 features by importance
- Combine: 3 methods (soft blend, hard threshold, adjustment ratio)
- Post-process: Isotonic regression calibration (OOF nested)

## Results
- CV MAE: [unknown] (worsened vs Phase 16)

## Lessons (CRITICAL)
- **Extreme samples too few for MLP**: At T=100, only ~3,786 extreme samples
  - 5-fold split → ~3,000 train per fold
  - MLP with 128-64-32 = ~12K parameters → severe overfitting risk
  - Validation extreme samples per fold: ~750, too few for reliable early stopping
- **Classification → regression pipeline is fragile**
  - Classifier errors compound with regression errors
  - False positives: normal samples get extreme MLP prediction → large error
  - False negatives: extreme samples get GBDT prediction → no improvement
- **Isotonic calibration assumption violated**
  - Isotonic assumes monotonic relationship between predicted and actual
  - Two-stage blend creates a non-monotonic prediction surface
- **Fundamental issue**: The problem may not have enough extreme samples to train a separate model
  - Better approach: improve feature engineering so GBDT can use proxy signals for extremes
  - Or: add extreme-value features that help GBDT approximate the tail better
