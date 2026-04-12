# Phase 23 Pre-EDA: 데이터 구조 근본적 재검증

## 스크립트
- `run_phase23_preeda.py` (Kaggle CPU 15분 내)

## 5개 EDA
1. **Sequence Structure**: autocorrelation, shuffle test, shift_hour monotonic, variance ratio → TRUE_SEQUENCE / WEAK / INDEPENDENT 판정
2. **Scenario Aggregate**: AMEX 스타일 aggregate(mean/std/max/p90) correlation, deviation/zscore → STRONG / WEAK 판정
3. **Feature Importance**: Phase 16 ckpt에서 importance 분석, zero-importance removal 후보
4. **Bin 9 특성**: shift_hour/layout_type 분포, key feature 차이, layout 집중도, scenario cluster 패턴
5. **Distribution Shift**: GroupKFold adversarial AUC, top shift features

## 출력
- `output/phase23_eda/01_autocorr.csv`, `01_conclusion.txt`
- `output/phase23_eda/02_scenario_agg.csv`, `02_deviation.csv`
- `output/phase23_eda/03_feature_importance.csv`, `03_removal_candidates.txt`
- `output/phase23_eda/04_bin9_hour.csv`, `04_bin9_features.csv`
- `output/phase23_eda/05_adversarial.csv`
- `output/phase23_eda/SUMMARY.txt`
