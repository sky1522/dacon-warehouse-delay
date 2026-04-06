# 데이콘 스마트 창고 출고 지연 예측 — Phase 7: 우승 전략 통합

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- 최고 Public MAE: 10.203 (Phase 3B)
- 최고 CV MAE: 8.786 (Phase 3B)
- 참고 코드: run_phase3b.py (최고 성능 코드, 이걸 기반으로 확장)
- Optuna best params (LGB): learning_rate=0.0129, num_leaves=185, max_depth=9, min_child_samples=80, reg_alpha=0.0574, reg_lambda=0.0042, feature_fraction=0.6005, bagging_fraction=0.7663, bagging_freq=1

## ⚠️ 필수 사항
- concat+sort 후 test 분리 시 원본 ID 순서 복원 필수
- assert (submission['ID'] == sample_sub['ID']).all()
- run_phase7.py 파일 작성만 할 것. 실행하지 말 것.
- float32로 메모리 최적화 (RAM 8GB 환경)
- 중간 진행상황 print 충분히 포함

## 핵심 전략 3가지 (경쟁자 코드 분석 + 우리 도메인 분석 기반)

### 전략 1: 샘플 가중치 (Sample Weight)
우리 Codex 분석에서 핵심 문제: 50분+ 극단값에서 MAE 33~134, 94% 과소예측.
극단값 행에 높은 가중치를 줘서 모델이 더 집중하게 함.

### 전략 2: 다양한 변환+목적함수 앙상블
같은 데이터를 다른 시각으로 보는 모델들을 섞으면 다양성 극대화.
지금까지 log1p+MAE만 사용했으므로 변환/목적함수 조합을 다양화.

### 전략 3: 새로운 시계열 피처 (Onset + Expanding Mean)
현재 상태뿐 아니라 "언제 상태 변화가 시작됐는가", "시나리오 전체 추세에서 현재가 얼마나 벗어났는가"를 표현.

## 수행 작업 (run_phase7.py)

### 1. 데이터 준비
- run_phase3b.py의 데이터 로드 + 시계열 피처(64개) + 인터랙션(8개) 로직 재사용
- layout_info 조인, implicit_timeslot 생성
- 모든 float64를 float32로 변환하여 메모리 절약
- 불필요한 중간 DataFrame은 del + gc.collect()로 즉시 해제

### 2. 신규 피처 추가 — Onset 피처 (상태 변화 시작점 추적)

시나리오 내에서 "언제 처음으로 특정 상태가 발생했는가"를 추적:

**충전 관련 onset:**
- charging_ever_started: 해당 시나리오에서 robot_charging > 0인 슬롯이 과거에 있었는지 (0/1)
- charging_start_idx: robot_charging이 처음 0보다 커진 슬롯 번호 (아직이면 -1)
- charging_steps_since_start: 충전 시작 이후 경과 슬롯 수
- charging_started_early: 충전이 슬롯 5 이전에 시작됐는지 (0/1)

**대기열 관련 onset:**
- queue_ever_started: charge_queue_length > 0인 슬롯이 과거에 있었는지
- queue_start_idx: 대기열이 처음 발생한 슬롯 번호

**혼잡 관련 onset:**
- congestion_ever_started: congestion_score > 0인 슬롯이 과거에 있었는지
- congestion_start_idx: 혼잡이 처음 발생한 슬롯 번호

구현: 각 피처에 대해 groupby(scenario_id) 내에서 shift(1)한 값의 cummax/cummin으로 계산. 미래 정보 사용하지 않도록 반드시 shift(1) 적용.

### 3. 신규 피처 추가 — Expanding Mean (누적 평균 + 이탈)

시나리오 시작부터 현재까지의 누적 통계와 현재값의 차이:

대상 컬럼 (핵심 15개):
order_inflow_15m, unique_sku_15m, avg_items_per_order, urgent_order_ratio,
heavy_item_ratio, robot_active, battery_mean, low_battery_ratio,
congestion_score, max_zone_density, pack_utilization, loading_dock_util,
charge_queue_length, fault_count_15m, avg_trip_distance

각 컬럼에 대해:
- {col}_expmean_prev: shift(1) 후 expanding().mean() — 과거 슬롯들의 평균
- {col}_delta_expmean: 현재값 - expmean_prev — 누적 평균 대비 이탈

총 30개 피처 추가.

### 4. 신규 피처 추가 — 비선형 임계값 피처

특정 임계값을 넘으면 지연이 급증하는 비선형 관계 표현:

- battery_mean_below_44: max(44.0 - battery_mean, 0) — 배터리 44% 이하 심각도
- low_battery_ratio_above_02: max(low_battery_ratio - 0.2, 0) — 저배터리 비율 20% 초과분
- pack_utilization_sq: pack_utilization ** 2 — 패킹 활용률 비선형 효과
- loading_dock_util_sq: loading_dock_util ** 2
- congestion_score_sq: congestion_score ** 2
- charge_pressure: (robot_charging + charge_queue_length) / (charger_count + 1)
- charge_pressure_sq: charge_pressure ** 2

### 5. 시간대별 위상 피처 (Phase Indicator)
- is_early_phase: implicit_timeslot <= 5 (0/1)
- is_mid_phase: 6 <= implicit_timeslot <= 15 (0/1)
- is_late_phase: implicit_timeslot >= 16 (0/1)
- time_frac: implicit_timeslot / 24.0
- time_remaining: 24 - implicit_timeslot
- time_frac_sq: time_frac ** 2

### 6. 샘플 가중치 함수 구현
```python
def build_sample_weight(y, time_idx):
    w = np.ones(len(y), dtype=np.float32)
    q90 = np.nanquantile(y, 0.90)
    q95 = np.nanquantile(y, 0.95)
    q99 = np.nanquantile(y, 0.99)
    w += 0.15 * (y >= q90).astype(np.float32)
    w += 0.30 * (y >= q95).astype(np.float32)
    w += 0.60 * (y >= q99).astype(np.float32)
    # 후반 타임슬롯 가중치
    if time_idx is not None:
        w += 0.08 * (time_idx / 24.0).astype(np.float32)
    return w
```

LightGBM: fit(X, y, sample_weight=w)
CatBoost: Pool(X, y, weight=w)
XGBoost: fit(X, y, sample_weight=w)

### 7. 4모델 앙상블 — 다양한 변환+목적함수

**모델 1: LightGBM + 변환 없음 + objective='mae'**
- Optuna best params 사용
- target_transform='none'
- sample_weight 적용

**모델 2: LightGBM + log1p + objective='huber' (alpha=0.9)**
- n_estimators=2000, learning_rate=0.03, num_leaves=128
- min_child_samples=60, subsample=0.9, colsample_bytree=0.85
- reg_alpha=0.05, reg_lambda=1.0
- sample_weight 적용

**모델 3: XGBoost + 변환 없음 + objective='reg:absoluteerror'**
- n_estimators=2000, learning_rate=0.03, max_depth=8
- min_child_weight=6, subsample=0.9, colsample_bytree=0.85
- reg_lambda=1.5, reg_alpha=0.05
- sample_weight 적용

**모델 4: CatBoost + log1p + loss_function='MAE'**
- iterations=2000, learning_rate=0.03, depth=8
- l2_leaf_reg=5.0, subsample=0.9
- sample_weight 적용

각 모델 scenario_id GroupKFold 5-Fold:
- OOF 예측 저장
- Fold별 MAE 출력
- 평균 CV MAE 출력

### 8. 앙상블 가중치 최적화
- 4개 OOF 예측으로 scipy.optimize.minimize
- 제약: 가중치 합 = 1, 각 >= 0
- 최적 가중치와 앙상블 CV MAE 출력

### 9. 결과 비교 테이블
=== Phase 7 결과 ===
신규 피처 수: onset N개 + expanding N개 + 비선형 N개 + 위상 N개 = 총 N개
총 피처 수: N개
모델별 CV MAE:
LGB raw+MAE:        X.XXXX
LGB log1p+Huber:    X.XXXX
XGB raw+MAE:        X.XXXX
CatBoost log1p+MAE: X.XXXX
앙상블 가중치: lgb_raw=X.XX, lgb_huber=X.XX, xgb=X.XX, cat=X.XX
앙상블 CV MAE: X.XXXX
Phase 3B 대비 개선: X.XXXX
샘플 가중치 효과:
가중치 적용 전: X.XXXX
가중치 적용 후: X.XXXX

### 10. 제출 파일
- output/submission_phase7.csv
- ID 순서 assert 검증

### 11. 피처 중요도
- 모델 1 (LGB raw+MAE) 기준 Top 30 시각화
- output/feature_importance_phase7.png
- 신규 피처 중 Top 30 진입 개수 출력

## 규칙
- 시각화 한글 폰트 적용 (한글 폰트가 없으면 영문으로 대체)
- 모든 MAE print 출력
- 중간 진행상황 print 충분히 포함
- n_estimators=2000, early_stopping_rounds=100
- 결과를 claude_results.md에 Phase 7 섹션 추가
- run_phase7.py 작성만 하고 실행하지 말 것
- 커밋 메시지: "feat: Phase 7 - sample weight + diverse ensemble + onset features"
- 커밋 + 푸시