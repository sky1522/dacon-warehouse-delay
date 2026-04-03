# 데이콘 스마트 창고 출고 지연 예측 — Phase 3B: 잔차 분석 기반 개선 + Optuna

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- 최고 Public MAE: 10.224 (Phase 2)
- CV MAE: 8.825 (앙상블)
- 핵심 문제: 50분+ 극단 지연(전체 8%)에서 MAE 33~134. 94%가 과소예측.
- 후반 슬롯(19-24)에서 에러 집중 (MAE 10.5 vs 초반 6.9)
- 참고 코드: run_phase2.py

## ⚠️ ID 순서 버그 방지 + 실행 관련
- concat+sort 후 test 분리 시 원본 ID 순서 복원 필수
- assert (submission['ID'] == sample_sub['ID']).all()
- run_phase3b.py 파일 작성만 할 것. 실행하지 말 것.

## 수행 작업 (run_phase3b.py)

### 1. 데이터 준비
- run_phase2.py와 동일한 로직 재사용 (데이터 로드 + 시계열 피처 + layout 조인)

### 2. 잔차 분석 기반 신규 피처 추가 (Codex 분석에서 발견)
아래 인터랙션 피처를 train/test 모두에 추가:
- orders_per_packstation = order_inflow_15m / pack_station_count
- pack_dock_pressure = pack_utilization * loading_dock_util
- dock_wait_pressure = outbound_truck_wait_min * loading_dock_util
- shift_load_pressure = prev_shift_volume * order_inflow_15m
- battery_congestion = low_battery_ratio * congestion_score
- storage_density_congestion = storage_density_pct * congestion_score
- battery_trip_pressure = low_battery_ratio * avg_trip_distance
- demand_density = order_inflow_15m * max_zone_density
(나눗셈 시 분모가 0이면 NaN 처리, LightGBM이 자체 처리)

### 3. 극단값 대응 — 듀얼 모델 전략
**모델 A: 기본 모델 (전체 데이터)**
- LightGBM, log1p target, objective=mae

**모델 B: 고지연 특화 모델 (타겟 >= 20분인 행만 학습)**
- LightGBM, log1p target, objective=mae
- 고지연 패턴에 집중하여 극단값 예측력 강화

**최종 예측 결합:**
- 모델 A 예측값이 20 미만이면 모델 A 사용
- 모델 A 예측값이 20 이상이면 모델 B 예측값 사용
- 임계값(20)도 CV 기준으로 최적화 (10, 15, 20, 25, 30 비교)

### 4. Optuna 튜닝 (모델 A만, 50 trials)
탐색 범위:
- num_leaves: 31~255
- max_depth: 4~12
- learning_rate: 0.01~0.1 (log uniform)
- n_estimators: 고정 3000, early_stopping_rounds=100
- min_child_samples: 10~100
- reg_alpha: 1e-3~10.0 (log uniform)
- reg_lambda: 1e-3~10.0 (log uniform)
- feature_fraction: 0.5~0.9
- bagging_fraction: 0.5~0.9
- bagging_freq: 1~7

목적함수: scenario_id GroupKFold 5-Fold MAE
매 10 trial마다 현재 best 출력

### 5. 최종 앙상블
- Optuna best params로 LightGBM 학습
- CatBoost, XGBoost도 기본 파라미터로 학습 (신규 피처 포함)
- 3모델 앙상블 가중치 최적화
- 듀얼 모델 전략 적용 여부도 CV 비교 후 결정

### 6. 결과 비교
=== Phase 3B 결과 ===
신규 피처 추가 효과: LGB CV MAE X.XXXX (vs Phase2 8.8508)
듀얼 모델 (최적 임계값): CV MAE X.XXXX
Optuna 최적 LGB: CV MAE X.XXXX (best params: {...})
최종 앙상블: CV MAE X.XXXX
Phase 2 앙상블: 8.8253

### 7. 제출 파일
- output/submission_phase3b.csv
- ID 순서 assert 검증

## 규칙
- 시각화 한글 폰트 적용
- 모든 MAE print 출력
- Optuna: optuna.logging.set_verbosity(optuna.logging.WARNING)
- 결과를 claude_results.md에 Phase 3B 섹션 추가
- run_phase3b.py 작성만 하고 실행하지 말 것
- 커밋 메시지: "feat: Phase 3B - residual-based features + dual model + Optuna"
- 커밋 + 푸시