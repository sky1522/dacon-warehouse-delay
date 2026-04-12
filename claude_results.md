# Phase 23 Track A: AMEX Aggregate + Saturation + Queueing

## EDA 결과 반영
- INDEPENDENT_SNAPSHOTS (autocorr 0.29) → sequence/lag feature 금지
- STRONG aggregate (improvement 0.21) → scenario aggregate 대거 추가
- Deviation corr ~0 → deviation feature 금지
- Bin 9 원인: pack/robot 포화 → saturation features
- Adversarial AUC 0.66 → layout-aware features

## 10-Step Pipeline
1. **핵심 변수 정의**: OPERATION_COLS (~50), LAYOUT_COLS (~13)
2. **Scenario Aggregate**: mean/std/min/max/median/p90/p10/range per scenario (AMEX 2위 스타일)
3. **Saturation Features**: pack/robot/dock 포화 indicators, margin, combined pressure (~15)
4. **Queueing Theory**: W_pack, W_robot, W_dock (rho/(1-rho)), Little's Law (~10)
5. **Layout-aware**: capacity ratios, layout type one-hot (~8)
6. **OOF Target Encoding**: layout_id, layout_type (GroupKFold, Bayesian smoothing)
7. **NaN Pattern**: count, ratio, scenario-level, critical column flags (~10)
8. **Feature Selection**: zero importance + bottom 10% removal
9. **5-Fold CV**: LGB Huber (GPU with CPU fallback), 5000 trees
10. **Save**: submission, OOF, checkpoint

## 출력
- `output/phase23_track_a_submission.csv`
- `output/phase23_track_a_oof.npy`, `_test.npy`
- `output/phase23_track_a_importance.csv`
- `output/phase23_track_a_ckpt.pkl`
