# 데이콘 스마트 창고 출고 지연 예측 — 결과 요약

## EDA 주요 발견 (notebooks/01_eda.ipynb)
1. **타겟 분포**: mean 18.96, median 9.03, skew 5.68, max 715.86 → 강한 우측 꼬리 분포, log1p 변환 시 정규 분포에 가까워짐
2. **타겟 상관관계 Top 5**: low_battery_ratio(+0.37), battery_mean(-0.36), robot_idle(-0.35), order_inflow_15m(+0.34), robot_charging(+0.32)
3. **timeslot 추이**: implicit_timeslot이 증가할수록 지연 시간 평균이 점진적으로 증가하는 경향
4. **layout_type**: layout_type별로 타겟 분포에 유의미한 차이 존재
5. **배터리 피처**: low_battery_ratio가 높을수록, battery_mean이 낮을수록 지연 증가 경향 뚜렷
6. **결측**: 86/94 컬럼에 결측 존재, 비율은 11.8~13.4%로 균일

## 베이스라인 모델 결과 (notebooks/02_baseline.ipynb)

### 피처 엔지니어링
- implicit_timeslot 생성
- layout_info.csv 조인 + layout_type 라벨 인코딩
- 정규화 레이아웃 피처 3개 (robot_per_area, charger_per_robot, packstation_per_robot)
- 수요/용량 비율 2개 (order_per_active_robot, sku_per_packstation)
- 배터리 인터랙션 2개 (battery_bottleneck, battery_spread)
- 결측 지시자 10개
- 총 피처 수: 122개

### 검증 전략
- scenario_id 기준 GroupKFold (5-Fold)

### CV MAE 결과

| 방법 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 평균 MAE |
|------|--------|--------|--------|--------|--------|---------|
| 원본 타겟 | 9.3171 | 9.1750 | 9.7470 | 8.6453 | 9.0662 | **9.1901** |
| log1p 타겟 | 9.3109 | 9.1580 | 9.7253 | 8.6566 | 9.0590 | **9.1820** |

### 결론
- **log1p 타겟이 근소하게 우수** (MAE 9.1820 vs 9.1901, 차이 0.0081)
- `output/submission_best.csv`는 log1p 방법으로 생성됨

### 생성된 파일
- `output/submission_baseline.csv` — 원본 타겟 예측
- `output/submission_best.csv` — log1p 타겟 예측 (best)
- `output/feature_importance_top30.png` — 피처 중요도 시각화
- `output/eda_*.png` — EDA 시각화 6개
