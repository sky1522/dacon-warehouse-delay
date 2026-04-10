# EDA: Target Distribution

## 기본 통계
- mean: 18.96, median: 9.03
- std: 37.94, skew: 5.68, kurtosis: ~64
- min: 0.0, max: 715.86
- log1p 변환 시 정규 분포에 근사

## Bin 분포 (10분위)
- Bin 0~8: MAE 2~10 수준 (총 MAE의 52%)
- **Bin 9 (상위 10%)**: MAE 40.92 = 전체 MAE의 **48%**
- Bin 9 target range: ~100~715

## 핵심 발견
- 전체 MAE 개선의 핵심은 Bin 9
- Bin 9 과소예측이 지배적 (100% underprediction in top 20 worst)
- GBDT prediction max ~120 (train max 715 대비 구조적 한계)

## 상관관계 Top 5
1. low_battery_ratio: +0.37
2. battery_mean: -0.36
3. robot_idle: -0.35
4. order_inflow_15m: +0.34
5. robot_charging: +0.32

## 시사점
- log1p/sqrt 변환으로 tail 압축 → 다양한 transform 앙상블
- Bin 9 공략은 피처 엔지니어링으로 (GBDT extrapolation 한계 인정)
