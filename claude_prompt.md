# 데이콘 스마트 창고 출고 지연 예측 — Phase 8: 경쟁자 피처 완전 적용 + 스태킹

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- 최고 Public MAE: 10.149 (Phase 7, 34위)
- 최고 CV MAE: 8.675 (Phase 7 앙상블)
- 피처: 245개
- 10위 진입 목표: Public 9.99 이하 (CV 약 8.5 이하 필요)
- 참고 코드: run_phase7.py (현재 최고 성능 코드, 이걸 기반으로 확장)

## ⚠️ 필수 사항
- concat+sort 후 test 분리 시 원본 ID 순서 복원 필수
- assert (submission['ID'] == sample_sub['ID']).all()
- run_phase8.py 파일 작성만 할 것. 실행하지 말 것.
- float32로 메모리 최적화 (Colab 12GB RAM 환경)
- del + gc.collect()로 중간 메모리 해제
- 중간 진행상황 print 충분히 포함

## 전략 개요
Phase 7 코드를 기반으로 3가지 추가:
A) 경쟁자가 쓰고 우리가 안 쓴 피처 50개+ 추가
B) Expanding 피처 확장 (std, max 추가)
C) 2단계 스태킹 (OOF → 메타 모델)

## 수행 작업 (run_phase8.py)

### 1. 데이터 준비
- run_phase7.py의 전체 로직 재사용:
  * 데이터 로드 + layout 조인 + 시계열 피처 64개
  * Phase 3B 인터랙션 8개
  * Onset 8개 + Expanding mean/delta 30개
  * 비선형 7개 + 위상 6개
- 여기에 아래 신규 피처 추가

### 2. 신규 피처 A: 경쟁자 코드에서 가져올 피처

**A-1) 로봇 상태 분해 (6개)**
- robot_total_state = robot_active + robot_idle + robot_charging
- robot_total_gap = robot_total_state - robot_total
- robot_active_share = robot_active / (robot_total_state + 1)
- robot_idle_share = robot_idle / (robot_total_state + 1)
- robot_charging_share = robot_charging / (robot_total_state + 1)
- charging_to_active_ratio = robot_charging / (robot_active + 1)

**A-2) 수요-용량 비율 (15개)**
- inflow_per_robot = order_inflow_15m / (robot_total + 1)
- inflow_per_pack_station = order_inflow_15m / (pack_station_count + 1)
- unique_sku_per_robot = unique_sku_15m / (robot_total + 1)
- unique_sku_per_pack_station = unique_sku_15m / (pack_station_count + 1)
- charge_queue_per_charger = charge_queue_length / (charger_count + 1)
- charging_per_charger = robot_charging / (charger_count + 1)
- congestion_per_width = congestion_score / (aisle_width_avg + 0.01)
- zone_density_per_width = max_zone_density / (aisle_width_avg + 0.01)
- order_per_sqm = order_inflow_15m / (floor_area_sqm + 1)
- dock_pressure = order_inflow_15m / (staff_on_floor + 1)
- fault_per_active = fault_count_15m / (robot_active + 1)
- collision_per_active = near_collision_15m / (robot_active + 1)
- blocked_per_active = blocked_path_15m / (robot_active + 1)
- congestion_per_active = congestion_score / (robot_active + 1)
- label_queue_per_pack = label_print_queue / (pack_station_count + 1)

**A-3) 복합 인터랙션 (12개)**
- demand_mass = order_inflow_15m * avg_package_weight_kg
- demand_mass_per_robot = demand_mass / (robot_total + 1)
- trip_load = order_inflow_15m * avg_trip_distance
- trip_load_per_robot = trip_load / (robot_total + 1)
- complexity_load = order_inflow_15m * unique_sku_15m
- complexity_load_per_pack = complexity_load / (pack_station_count + 1)
- congestion_x_lowbat = congestion_score * low_battery_ratio
- battery_pressure = low_battery_ratio * robot_active
- queue_wait_pressure = charge_queue_length * avg_charge_wait
- dock_pack_pressure = loading_dock_util * pack_utilization
- staging_pack_pressure = staging_area_util * pack_utilization
- charge_pressure = (robot_charging + charge_queue_length) / (charger_count + 1)

**A-4) layout 밀도 피처 (8개)**
- warehouse_volume = floor_area_sqm * ceiling_height_m
- intersection_density = intersection_count / (floor_area_sqm + 1)
- pack_station_density = pack_station_count / (floor_area_sqm + 1)
- charger_density = charger_count / (floor_area_sqm + 1)
- robot_density_layout = robot_total / (floor_area_sqm + 1)
- movement_friction = intersection_count / (aisle_width_avg + 0.01)
- layout_compact_x_dispersion = layout_compactness * zone_dispersion
- one_way_friction = one_way_ratio * intersection_count / (aisle_width_avg + 0.01)

**A-5) missing indicator (3개)**
- n_missing_all = 원본 수치 컬럼 중 NaN 개수 (행별)
- n_missing_dynamic = 동적 피처(order_inflow, battery 등 시계열성 컬럼) 중 NaN 개수
- missing_ratio = n_missing_all / 전체 수치 컬럼 수

**A-6) rolling max + 편차 (시계열 보강, 10개)**
sequence_cols 중 상위 5개(order_inflow_15m, battery_mean, congestion_score, pack_utilization, loading_dock_util)에 대해:
- {col}_rollmax3_prev: shift(1) 후 rolling(3).max()
- {col}_dev_rollmax3: 현재값 - rollmax3_prev

### 3. 신규 피처 B: Expanding 확장 (20개)

Phase 7에서 expanding mean이 압도적이었으므로 확장:
- 핵심 10개 컬럼에 대해:
  order_inflow_15m, battery_mean, congestion_score, pack_utilization,
  loading_dock_util, robot_active, low_battery_ratio, avg_trip_distance,
  unique_sku_15m, max_zone_density
- 각 컬럼에:
  * {col}_expstd_prev: shift(1) → expanding().std()
  * {col}_expmax_prev: shift(1) → expanding().max()

### 4. 피처 수 확인
기존 245 + 로봇상태 6 + 수요용량 15 + 인터랙션 12 + layout밀도 8 + missing 3 + rolling편차 10 + expanding확장 20 = 약 319개
신규 피처 수와 전체 피처 수 출력

### 5. Level 1: 6모델 학습 (scenario_id GroupKFold 5-Fold)

샘플 가중치 함수 (Phase 7과 동일):
```python
def build_sample_weight(y, time_idx):
    w = np.ones(len(y), dtype=np.float32)
    q90 = np.nanquantile(y, 0.90)
    q95 = np.nanquantile(y, 0.95)
    q99 = np.nanquantile(y, 0.99)
    w += 0.15 * (y >= q90).astype(np.float32)
    w += 0.30 * (y >= q95).astype(np.float32)
    w += 0.60 * (y >= q99).astype(np.float32)
    if time_idx is not None:
        w += 0.08 * (time_idx / 24.0).astype(np.float32)
    return w
```

6개 모델 (다양성 극대화):

**모델 1: LightGBM raw+MAE (Optuna params)**
- target_transform='none', objective='mae'
- learning_rate=0.0129, num_leaves=185, max_depth=9
- min_child_samples=80, reg_alpha=0.0574, reg_lambda=0.0042
- feature_fraction=0.6005, bagging_fraction=0.7663, bagging_freq=1
- n_estimators=2000, early_stopping_rounds=100
- sample_weight 적용

**모델 2: LightGBM log1p+Huber**
- target_transform='log1p', objective='huber', alpha=0.9
- n_estimators=2000, learning_rate=0.03, num_leaves=128
- min_child_samples=60, subsample=0.9, colsample_bytree=0.85
- reg_alpha=0.05, reg_lambda=1.0
- sample_weight 적용

**모델 3: LightGBM sqrt+MAE (신규)**
- target_transform='sqrt', objective='mae'
- n_estimators=2000, learning_rate=0.03, num_leaves=96
- min_child_samples=80, subsample=0.9, colsample_bytree=0.85
- reg_alpha=0.1, reg_lambda=1.5
- sample_weight 적용

**모델 4: XGBoost raw+MAE**
- target_transform='none', objective='reg:absoluteerror'
- n_estimators=2000, learning_rate=0.03, max_depth=8
- min_child_weight=6, subsample=0.9, colsample_bytree=0.85
- reg_lambda=1.5, reg_alpha=0.05
- sample_weight 적용

**모델 5: CatBoost log1p+MAE**
- target_transform='log1p', loss_function='MAE'
- iterations=2000, learning_rate=0.03, depth=8
- l2_leaf_reg=5.0, subsample=0.9
- sample_weight 적용

**모델 6: CatBoost raw+MAE (신규)**
- target_transform='none', loss_function='MAE'
- iterations=2000, learning_rate=0.03, depth=6
- l2_leaf_reg=3.0, subsample=0.85
- sample_weight 적용

각 모델에서:
- OOF 예측값 저장 (train 전체에 대한 out-of-fold 예측)
- test 예측값 저장 (5-Fold 평균)
- Fold별 MAE + 평균 CV MAE 출력

### 6. Level 1 앙상블 (단순 가중 평균 — 비교 기준)
- 6개 OOF로 scipy.optimize.minimize 가중치 최적화
- 가중 평균 CV MAE 출력 (Phase 7과 비교용)

### 7. Level 2: 스태킹 메타 모델
**메타 피처 구성:**
- 6개 모델의 OOF 예측값 (6개 컬럼)
- implicit_timeslot (시간 정보)
- 원본 핵심 피처 5개: order_inflow_15m, battery_mean, robot_active, pack_utilization, congestion_score

총 메타 피처: 12개

**메타 모델: Ridge Regression**
- sklearn.linear_model.Ridge, alpha=1.0
- scenario_id GroupKFold 5-Fold
- 메타 OOF MAE 출력

**메타 모델 대안: LightGBM (소규모)**
- n_estimators=200, learning_rate=0.05, num_leaves=16, max_depth=4
- 과적합 방지를 위해 작은 모델
- scenario_id GroupKFold 5-Fold
- 메타 OOF MAE 출력

**더 나은 메타 모델 선택 후 최종 test 예측**

### 8. 결과 비교
=== Phase 8 결과 ===
신규 피처: 경쟁자 A(54개) + Expanding 확장 B(20개) = 74개
총 피처 수: ~319개
Level 1 모델별 CV MAE:
LGB raw+MAE:        X.XXXX
LGB log1p+Huber:    X.XXXX
LGB sqrt+MAE:       X.XXXX
XGB raw+MAE:        X.XXXX
CatBoost log1p+MAE: X.XXXX
CatBoost raw+MAE:   X.XXXX
Level 1 가중 평균 앙상블: X.XXXX
Level 2 Ridge 스태킹:    X.XXXX
Level 2 LGB 스태킹:      X.XXXX
최종 선택:               X.XXXX
Phase 7 대비 개선:       X.XXXX

### 9. 제출 파일
- 최종 최고 CV 모델로 output/submission_phase8.csv 생성
- ID 순서 assert 검증

### 10. 피처 중요도
- Level 1 모델 1 (LGB raw+MAE) 기준 Top 30 시각화
- output/feature_importance_phase8.png
- 경쟁자 피처 중 Top 30 진입 개수 별도 출력

## 규칙
- 시각화: 한글 폰트 없으면 영문으로 대체 (Colab 환경)
- 모든 MAE print 출력
- 중간 진행상황 print 충분히 포함
- 결과를 claude_results.md에 Phase 8 섹션 추가
- run_phase8.py 작성만 하고 실행하지 말 것
- 커밋 메시지: "feat: Phase 8 - competitor features + stacking"
- 커밋 + 푸시