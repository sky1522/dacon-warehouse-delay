# run_phase20_clean.py review

## Findings

1. High - The adversarial stage will fail at runtime because `roc_auc_score` is used but never imported.
   - Evidence: imports at `run_phase20_clean.py:15`, usage at `run_phase20_clean.py:936`.
   - Impact: `adv_auc`, `adv_weight`, and `holdout_mask` are never produced.

2. High - The reported `Holdout MAE` is not an independent ensemble metric.
   - Evidence: ensemble weights are optimized on the full OOF matrix at `run_phase20_clean.py:1151-1159`, then evaluated on the holdout subset at `run_phase20_clean.py:1170-1173`.
   - Impact: if this number is used for final model selection, it is optimistically biased because holdout targets already influenced the ensemble weights.

3. Medium - Adversarial validation ignores grouping even though the main CV is grouped by `layout_id` and the shift is heavily layout-driven.
   - Evidence: main training uses `StratifiedGroupKFold(..., groups=layout_id)` at `run_phase20_clean.py:960-962`, while adversarial validation uses plain `StratifiedKFold` at `run_phase20_clean.py:921-929`.
   - Data check: train/test share 50 `layout_id` values and 0 `scenario_id` values.
   - Impact: the same layout can appear in adversarial train and valid splits, which can inflate the AUC 0.989 and distort `adv_weight`.

4. Medium - Median imputation is fit once on the full training set before any CV split, so OOF metrics include mild preprocessing leakage.
   - Evidence: median fill happens at `run_phase20_clean.py:90-96`, feature generation starts immediately after at `run_phase20_clean.py:101-110`, and the main CV split is only created later at `run_phase20_clean.py:960-962`.
   - Impact: OOF/CV results are a little more optimistic than a fold-local preprocessing pipeline.

5. Medium - The final submission is not a full-train refit.
   - Evidence: all models train over 5 folds from `run_phase20_clean.py:961-962` and `run_phase20_clean.py:992-1108`, and submission uses the average of fold predictions at `run_phase20_clean.py:1177-1181`.
   - Impact: if the intended policy is "train once on 100 percent of train before submit", that is not what this script does.

## Checklist

1. `fillna(median)` timing
   - Yes. It happens before Phase 13s1 feature generation.
   - References: `run_phase20_clean.py:75-99`, feature generation starts at `run_phase20_clean.py:104`.

2. Adversarial weight formula
   - The implementation matches the stated formula.
   - References: `run_phase20_clean.py:939-941`
   - Formula: `proba / (1 - proba + 1e-6)` then `clip(0.1, 10)` then normalize to mean 1.
   - Caveat: the formula is correct, but the input probabilities may be biased because the adversarial split is row-level.

3. `sample_weight` multiplication order
   - Correct. It is `base * adv`.
   - References: `run_phase20_clean.py:964-973`
   - The base weight is built first, then multiplied by `adv_w`.

4. Whether adversarial holdout is independent from the main CV folds
   - It is not directly tied to the main training folds.
   - `holdout_mask` is built from adversarial OOF probabilities at `run_phase20_clean.py:917-948`.
   - Main model folds are built separately at `run_phase20_clean.py:960-962`.
   - Caveat: the split logic is still not fully clean because adversarial validation does not respect layout groups.

5. MLP loss tensor shape for `(rmse + mae) / 2`
   - The loss formula itself is correct at `run_phase20_clean.py:1080-1083`.
   - The target is 1D at `run_phase20_clean.py:981`, while the model output is shape `(batch, 1)` at `run_phase20_clean.py:1086-1090`.
   - I could not execute TensorFlow in this environment, so I could not confirm the runtime shape handling.
   - Safer implementation: reshape targets to `(-1, 1)` or expand `y_true` inside the custom loss.

6. Whether `Holdout MAE` is used as the final decision metric
   - Not automatically in code. It is only printed.
   - The actual ensemble optimization objective is full-OOF MAE at `run_phase20_clean.py:1151-1159`.
   - `holdout_mae` is printed later at `run_phase20_clean.py:1170-1175` and `run_phase20_clean.py:1193`.
   - Caveat: if a human uses this number for phase selection, the bias from finding 2 still applies.

7. Whether final submission retrains on all train or only uses 80 percent folds
   - It uses fold models trained on about 80 percent each.
   - The final submission is the average of the 5 fold predictions.
   - There is no explicit full-train refit stage.

## Open Questions / Residual Risk

- MLP does not use `sample_weight` at `run_phase20_clean.py:1103`, even though Part 6 is presented as training with adversarial weights.
- TabNet also does not use `sample_weight` at `run_phase20_clean.py:1126`.
- Syntax is valid by `ast.parse`.
- I did not run the full script, and TensorFlow is not installed in the current environment.
