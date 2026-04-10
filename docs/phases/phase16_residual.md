# Phase 16 Residual Analysis

## Hypothesis
Detailed residual analysis of Phase 16 ensemble will reveal where to focus Phase 17+ improvements.

## Implementation
- 8 analysis parts: target bin, layout, position, feature values, worst predictions, scenario variability
- Automated Phase 17 direction diagnosis

## Results
- Phase 16 OOF MAE: 8.4403
- **Bin 9 MAE: 40.92** (전체 MAE의 48%)
- Hard layout top 5 MAE: 30.80 (systematic bias +15~18)
- Position: first 3 MAE 높음 (lag NaN), last 3 MAE 높음 (underprediction)
- Top 20 worst: 100% 과소예측, pack_utilization=1.0 + order_inflow 상승 패턴

## Lessons
- Bin 9이 MAE의 절반 차지 — 여기를 개선해야 순위 상승
- GBDT prediction max ~120 (train max 715 대비 구조적 한계)
- → Phase 17 (explosion features), Phase 18 (two-stage) 근거
