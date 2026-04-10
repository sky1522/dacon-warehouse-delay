# Progress Tracker

## Current Best
- **Phase 16**: CV 8.4403, Public 9.8795, 5위
- Cross-phase blend: Public 9.86105, 5위

## Phase Summary

| Phase | Status | CV | Public | One-liner |
|-------|--------|-----|--------|-----------|
| 1 | Done | 8.8508 | 10.249 | Time-series features (64) |
| 2 | Done | 8.8253 | - | 3-model ensemble |
| 3A/3B | Done | [unknown] | - | Feature selection + Optuna |
| 5 | Done | [unknown] | - | GroupBy 300 + target lag |
| 6 | Done | [unknown] | - | 22 domain features |
| 7 | Done | [unknown] | - | Sample weights + 4-model |
| 8 | Done | [unknown] | - | 54 competitor features + stacking |
| 9 | Done | [unknown] | - | Multi-seed + Optuna retune |
| 10 | Done | [unknown] | - | MLP + TabNet (8 models) |
| 11 | Failed | [unknown] | - | Adversarial + Transformer |
| 12a | Done | [unknown] | - | 42 queueing theory features |
| 13s1 | Done | 8.5668 | 10.008 | StratifiedGroupKFold(layout_id) |
| 13s3 | Failed | [unknown] | - | Tail weight (tree limitation) |
| 13s5a | Failed | [unknown] | - | Bin9 classifier OOF |
| 14 | Failed | [unknown] | - | GRU (raw features only) |
| 15 | Done | [unknown] | - | 1000+ agg features → top 200 |
| 15B | Done | [unknown] | - | TabNet retrained |
| 16 | **Best** | 8.4403 | 9.8795 | 2nd-order FE (~700 features) |
| 17 | Pending | [unknown] | - | Explosion + hardness + CV² |
| 18 | Failed | [unknown] | - | Two-stage hurdle (MLP extreme) |
| **20** | **Next** | - | - | Clean preprocessing + adv weight |

## Next Steps
1. Seed 777 학습 완료 대기
2. Phase 20 pre-EDA 실행 (run_phase20_eda.py)
3. Phase 20 본 학습 (run_phase20_clean.py)
