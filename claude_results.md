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

## Phase 1: 시계열 피처 추가 (run_phase1.py)

### 추가 피처
- Lag 피처 (1,2,3슬롯): 8개 핵심 피처 x 3 = 24개
- Rolling 피처 (window 3,5): mean + std = 8 x 4 = 32개
- Diff 피처: 4개
- Cumsum 피처: 4개
- **시계열 피처 합계: 64개**, 총 피처: 186개

### CV MAE: 8.8508 (Public MAE: 10.249, 14위/249명)

## Phase 2: 멀티모델 앙상블 (run_phase2.py)

### 모델별 CV MAE

| 모델 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 평균 MAE |
|------|--------|--------|--------|--------|--------|---------|
| LightGBM | 8.9732 | 8.8579 | 9.3478 | 8.3185 | 8.7567 | **8.8508** |
| CatBoost (depth=6) | 9.0069 | 8.8770 | 9.3985 | 8.3601 | 8.7651 | **8.8815** |
| XGBoost (depth=6) | 8.9784 | 8.8997 | 9.4738 | 8.4118 | 8.7994 | **8.9126** |

### 앙상블 결과
- **최적 가중치**: LightGBM=0.52, CatBoost=0.33, XGBoost=0.15
- **Ensemble CV MAE: 8.8253**
- Phase 1 대비 개선: 8.8508 → 8.8253 (0.0255 개선)
- 베이스라인 대비 개선: 9.1820 → 8.8253 (0.3567 개선)

### 버그 수정
- test 행 순서 불일치 버그 수정 (original_idx 보존 후 복원)
- ID 순서 assert 검증 통과

### 생성된 파일
- `output/submission_phase2.csv` — 앙상블 예측
- `output/feature_importance_phase2.png` — 피처 중요도 시각화

## Phase 3A: 일반화 강화 (run_phase3a.py)

### 실험 내용
1. **피처 선택**: 중요도 하위 30% 피처 제거
2. **정규화 강화**: LGB/Cat/XGB 모두 정규화 파라미터 강화
3. **layout 파생 피처**: robot_per_packstation, charger_density, intersection_density, robot_compactness, dispersion_robot
4. **layout GroupKFold**: unseen layout 시뮬레이션 검증

### 결과 (실행 후 업데이트 필요)
- 실험 0 (기준선): LGB CV MAE ?.????
- 실험 1 (피처 선택): LGB CV MAE ?.????
- 실험 2 (정규화 강화): LGB CV MAE ?.????
- 실험 3 (layout 피처): LGB CV MAE ?.????
- 실험 4 (layout GKF): LGB CV MAE ?.????
- 최종 앙상블 CV MAE: ?.????
- Phase 2 앙상블: 8.8253

### 생성된 파일
- `output/submission_phase3a.csv` — Phase 3A 앙상블 예측
- `output/feature_importance_phase3a.png` — 피처 중요도 시각화

## Phase 3B: 잔차 분석 기반 개선 + Optuna (run_phase3b.py)

### 실험 내용
1. **잔차 기반 신규 피처 8개**: orders_per_packstation, pack_dock_pressure, dock_wait_pressure, shift_load_pressure, battery_congestion, storage_density_congestion, battery_trip_pressure, demand_density
2. **듀얼 모델 전략**: 기본 모델 + 고지연 특화 모델 (임계값 최적화)
3. **Optuna 튜닝**: LightGBM 50 trials
4. **3모델 앙상블**: Optuna LGB + CatBoost + XGBoost

### 결과 (실행 후 업데이트 필요)
- 신규 피처 효과: LGB CV MAE ?.???? (vs Phase2 8.8508)
- 듀얼 모델: CV MAE ?.????
- Optuna 최적 LGB: CV MAE ?.????
- 최종 앙상블 CV MAE: ?.????
- Phase 2 앙상블: 8.8253

### 생성된 파일
- `output/submission_phase3b.csv` — Phase 3B 앙상블 예측
- `output/feature_importance_phase3b.png` — 피처 중요도 시각화

## Phase 5: Kaggle 우승 전략 적용 (run_phase5.py)

### 실험 내용
1. **대량 GroupBy 피처**: 4 그룹키 × 15 집계대상 × 5 통계 = 300개 피처 생성
2. **타겟 Lag 피처**: target_lag1~3, target_roll3_mean, target_cummax, target_cummean
3. **피처 선별**: 중요도 상위 300개 선택
4. **모델 A**: LightGBM (Optuna params, GroupBy 피처, 타겟lag 제외)
5. **모델 B**: LightGBM (타겟 Lag 포함, test 재귀 예측)
6. **CatBoost**: 모델 A 피처
7. **3모델 앙상블**: 가중치 최적화

### 결과 (실행 후 업데이트 필요)
- 모델 A: CV MAE ?.????
- 모델 B: CV MAE ?.???? (train 실제값 기준, 낙관적)
- CatBoost: CV MAE ?.????
- 최종 앙상블 CV MAE: ?.????
- Phase 3B 앙상블: 8.786

### 생성된 파일
- `output/submission_phase5.csv` — Phase 5 앙상블 예측
- `output/feature_importance_phase5.png` — 피처 중요도 시각화

## Phase 6: 도메인 지식 기반 피처 + Optuna 재튜닝 (run_phase6.py)

### 실험 내용
1. **도메인 기반 신규 피처 22개**:
   - A) 체인 병목 감지 (5개): picking_packing_gap, packing_shipping_gap, chain_pressure, picking_bottleneck, shipping_bottleneck
   - B) 로봇 가용 용량 (5개): available_capacity, charging_return_ratio, robot_shortage, effective_robot_ratio, robot_demand_balance
   - C) 누적 피로도 (5개): scenario_progress, battery_drain_rate, congestion_acceleration, late_scenario_flag, fatigue_index
   - D) 주문×창고 궁합 (5개): complex_in_narrow, urgent_pack_pressure, heavy_height_penalty, sku_per_intersection, order_density_per_area
   - E) 종합 위험도 (2개): risk_score, capacity_stress
2. **Optuna 재튜닝**: 50 trials (도메인 피처 포함)
3. **타겟 변환 비교**: log1p / sqrt / raw+mae / raw+huber
4. **2모델 앙상블**: LightGBM + CatBoost 가중치 최적화

### 결과 (실행 후 업데이트 필요)
- 도메인 피처 추가 (기존 params): LGB CV MAE ?.???? (vs Phase3B 8.7908)
- Optuna 재튜닝 후: LGB CV MAE ?.????
- 최적 타겟 변환: [?] CV MAE ?.????
- LightGBM 최종: CV MAE ?.????
- CatBoost: CV MAE ?.????
- 앙상블: CV MAE ?.????
- Phase 3B 대비 개선: ?.????

### 생성된 파일
- `output/submission_phase6.csv` — Phase 6 앙상블 예측
- `output/feature_importance_phase6.png` — 피처 중요도 시각화

## Phase 7: 우승 전략 통합 (run_phase7.py)

### 핵심 전략
1. **샘플 가중치**: q90/q95/q99 극단값에 높은 가중치 + 후반 타임슬롯 가중치
2. **다양한 변환+목적함수 앙상블**: 4모델 다양성 극대화
3. **신규 시계열 피처**: Onset(8개) + Expanding Mean(30개) + 비선형 임계값(7개) + 시간 위상(6개) = 51개

### 4모델 구성
- 모델 1: LightGBM raw+MAE (Optuna params)
- 모델 2: LightGBM log1p+Huber (alpha=0.9)
- 모델 3: XGBoost raw+MAE
- 모델 4: CatBoost log1p+MAE (depth=8)

### 결과 (실행 후 업데이트 필요)
- 샘플 가중치 효과: 적용 전 ?.???? → 적용 후 ?.????
- LGB raw+MAE: CV MAE ?.????
- LGB log1p+Huber: CV MAE ?.????
- XGB raw+MAE: CV MAE ?.????
- CatBoost log1p+MAE: CV MAE ?.????
- 앙상블: CV MAE ?.????
- Phase 3B 대비 개선: ?.????

### 생성된 파일
- `output/submission_phase7.csv` — Phase 7 앙상블 예측
- `output/feature_importance_phase7.png` — 피처 중요도 시각화

## Phase 8: 경쟁자 피처 완전 적용 + 스태킹 (run_phase8.py)

### 핵심 전략
1. **경쟁자 피처 54개**: 로봇상태분해(6) + 수요용량비율(15) + 복합인터랙션(12) + layout밀도(8) + missing(3) + rolling편차(10)
2. **Expanding 확장 20개**: 핵심 10개 컬럼에 expanding std + max
3. **2단계 스태킹**: 6모델 OOF → Ridge/LGB 메타 모델

### Level 1: 6모델
- 모델 1: LightGBM raw+MAE (Optuna params)
- 모델 2: LightGBM log1p+Huber
- 모델 3: LightGBM sqrt+MAE (신규)
- 모델 4: XGBoost raw+MAE
- 모델 5: CatBoost log1p+MAE
- 모델 6: CatBoost raw+MAE (신규)

### Level 2: 스태킹 메타 모델
- Ridge (alpha=1.0) + LGB 소규모 (num_leaves=16)
- 메타 피처: 6 OOF + implicit_timeslot + 핵심 5개 = 12개

### 결과 (실행 후 업데이트 필요)
- Level 1 모델별 CV MAE: 각 ?.????
- Level 1 가중 평균: CV MAE ?.????
- Ridge 스태킹: CV MAE ?.????
- LGB 스태킹: CV MAE ?.????
- 최종 선택: ?.????
- Phase 7 대비 개선: ?.????

### 생성된 파일
- `output/submission_phase8.csv` — Phase 8 최종 예측
- `output/feature_importance_phase8.png` — 피처 중요도 시각화

## Phase 9: Optuna 재튜닝 + Seed Blending + 스태킹 (run_phase9.py)

### 핵심 전략
1. **Optuna 재튜닝**: 319개 피처 기준 30 trials (기존은 194개 피처 기준)
2. **Multi-Seed Blending**: 6모델 × 3 seeds (42, 123, 777) → seed 평균으로 안정성 향상
3. **Level 2 LGB 스태킹**: 6 OOF + 핵심 6개 원본 피처 = 12개 메타 피처

### 6모델 구성 (Phase 8 동일 + Optuna NEW params)
- 모델 1: LGB raw+MAE (Optuna NEW)
- 모델 2: LGB log1p+Huber
- 모델 3: LGB sqrt+MAE
- 모델 4: XGB raw+MAE
- 모델 5: CatBoost log1p+MAE
- 모델 6: CatBoost raw+MAE

### 결과 (실행 후 업데이트 필요)
- Optuna 재튜닝: ?.???? (기존 8.7494)
- 모델별 seed-averaged CV MAE: 각 ?.????
- Level 1 가중 평균: ?.????
- Level 2 LGB 스태킹: ?.????
- Phase 8 대비 개선: ?.????

### 생성된 파일
- `output/submission_phase9.csv` — Phase 9 최종 예측
- `output/feature_importance_phase9.png` — 피처 중요도 시각화

## Phase 10: NN 추가 (Keras MLP + TabNet) + 8모델 스태킹 (run_phase10.py)

### 핵심 전략
- 트리 모델만 6개 → NN 2개(Keras MLP + TabNet) 추가로 다양성 극대화
- 8모델 스태킹으로 CV-Public 갭 축소 노림

### 8모델 구성
- 트리 6개: LGB raw+MAE, LGB log1p+Huber, LGB sqrt+MAE, XGB raw+MAE, Cat log1p+MAE, Cat raw+MAE
- NN 2개: Keras MLP (512-256-128-64, log1p), TabNet (n_d=32, n_a=32, entmax)

### 결과 (실행 후 업데이트 필요)
- 트리 6모델: 각 ?.????
- Keras MLP: ?.????
- TabNet: ?.????
- Level 1 트리만: ?.????
- Level 1 트리+NN: ?.????
- Level 2 LGB 스태킹: ?.????
- Phase 8 대비 개선: ?.????

### 생성된 파일
- `output/submission_phase10.csv` — Phase 10 최종 예측
- `output/feature_importance_phase10.png` — 피처 중요도 시각화
