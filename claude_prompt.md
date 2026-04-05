# 데이콘 스마트 창고 출고 지연 예측 — Phase 6: 도메인 지식 기반 피처 + Optuna

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- 최고 Public MAE: 10.203 (Phase 3B)
- 최고 CV MAE: 8.786 (Phase 3B, LGB+Cat 앙상블)
- 피처: 194개 (기존 122 + 시계열 64 + 인터랙션 8)
- Optuna best params: learning_rate=0.0129, num_leaves=185, max_depth=9, min_child_samples=80, reg_alpha=0.0574, reg_lambda=0.0042, feature_fraction=0.6005, bagging_fraction=0.7663, bagging_freq=1
- 참고 코드: run_phase3b.py (가장 좋은 Public 점수의 코드)
- Phase 5의 재귀 예측은 실패 → 재귀 예측 사용하지 말 것
- Phase 5의 GroupBy 300개 피처도 CV 개선 없었음 → 대량 groupby 대신 도메인 기반 정밀 피처

## ⚠️ 필수 사항
- concat+sort 후 test 분리 시 원본 ID 순서 복원 필수
- assert (submission['ID'] == sample_sub['ID']).all()
- run_phase6.py 파일 작성만 할 것. 실행하지 말 것.
- 중간 진행상황 print 충분히 포함

## 핵심 전략: 도메인 지식 기반 피처 엔지니어링
스마트 창고 출고 프로세스: 주문접수 → 피킹(로봇) → 패킹(포장) → 스테이징 → 트럭 적재 → 출고
지연은 이 체인의 병목에서 발생. 각 단계의 수요/용량 불균형과 연쇄 반응을 피처로 표현.

## 수행 작업 (run_phase6.py)

### 1. 데이터 준비
- run_phase3b.py의 데이터 로드 + 시계열 피처(64개) + 기존 인터랙션(8개) 로직 재사용
- layout_info 조인 포함
- implicit_timeslot 생성
- 기존 194개 피처를 베이스로 사용

### 2. 도메인 기반 신규 피처 추가 (train/test 모두)

**A) 체인 병목 감지 피처 (프로세스 단계 간 불균형)**
- picking_packing_gap = robot_utilization - pack_utilization
- packing_shipping_gap = pack_utilization - loading_dock_util
- chain_pressure = order_inflow_15m / (pack_station_count * (loading_dock_util + 0.01))
- picking_bottleneck = robot_utilization * (1 - pack_utilization)
- shipping_bottleneck = pack_utilization * loading_dock_util

**B) 로봇 가용 용량 피처**
- available_capacity = robot_idle - (robot_total * low_battery_ratio)
- charging_return_ratio = robot_charging / (robot_total + 1)
- robot_shortage = order_inflow_15m / (robot_idle + 1)
- effective_robot_ratio = (robot_active - robot_charging) / (robot_total + 1)
- robot_demand_balance = robot_active / (order_inflow_15m + 1)

**C) 누적 피로도 피처 (시나리오 진행에 따른 상태 변화)**
- scenario_progress = implicit_timeslot / 24.0
- battery_drain_rate: battery_mean의 diff1을 이미 계산했으면 재사용, 아니면 생성
- congestion_acceleration: congestion_score의 diff1 (혼잡 증가 속도)
- late_scenario_flag = (implicit_timeslot >= 19).astype(int)
- fatigue_index = scenario_progress * (1 - battery_mean/100) (진행률 × 배터리 소진)

**D) 주문 특성 × 창고 구조 궁합 피처**
- complex_in_narrow = avg_items_per_order / (aisle_width_avg + 0.01)
- urgent_pack_pressure = urgent_order_ratio * order_inflow_15m / (pack_station_count + 1)
- heavy_height_penalty = heavy_item_ratio * racking_height_avg_m
- sku_per_intersection = unique_sku_15m / (intersection_count + 1)
- order_density_per_area = order_inflow_15m / (floor_area_sqm + 1) * 10000

**E) 종합 위험도 지표**
- risk_score = (low_battery_ratio * 0.3 + congestion_score/100 * 0.3 + pack_utilization * 0.2 + loading_dock_util * 0.2)
- capacity_stress = (robot_utilization + pack_utilization + loading_dock_util) / 3

나눗셈 시 분모에 작은 값(0.01 또는 1) 더해서 ZeroDivision 방지.
NaN은 LightGBM이 자체 처리하므로 그대로 둘 것.
생성된 피처 수와 이름 목록 출력.

### 3. LightGBM 단독 CV (도메인 피처 효과 확인)
- Phase 3B Optuna params 사용
- scenario_id GroupKFold 5-Fold, log1p target
- CV MAE 출력
- "도메인 피처 추가 전(Phase 3B): 8.7908 → 추가 후: X.XXXX" 출력

### 4. Optuna 재튜닝 (도메인 피처 포함, 50 trials)
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

목적함수: scenario_id GroupKFold 5-Fold MAE (log1p → expm1)
optuna.logging.set_verbosity(optuna.logging.WARNING)
매 10 trial마다 best MAE 출력
sampler=optuna.samplers.TPESampler(seed=42)

### 5. 타겟 변환 비교 (Optuna best params로)
- log1p (현재)
- sqrt
- 변환 없음 + objective='mae'
- 변환 없음 + objective='huber' (alpha=10)
각각 5-Fold CV MAE 비교 출력

### 6. 최종 앙상블
- 최적 변환 + Optuna best params로 LightGBM 학습
- CatBoost 학습 (같은 피처, 기본 params)
- 2모델 앙상블 가중치 최적화
- 최종 CV MAE 출력

### 7. 결과 비교 테이블
=== Phase 6 결과 ===
도메인 피처 추가 후 (기존 params):  LGB CV MAE X.XXXX (vs Phase3B 8.7908)
Optuna 재튜닝 후:                   LGB CV MAE X.XXXX
최적 타겟 변환:                     [방법명] CV MAE X.XXXX
LightGBM 최종:                      CV MAE X.XXXX
CatBoost:                           CV MAE X.XXXX
앙상블:                             CV MAE X.XXXX
Phase 3B 대비 개선:                 X.XXXX
총 피처 수:                         N개
도메인 피처 중 중요도 Top 30 진입:  N개

### 8. 제출 파일
- output/submission_phase6.csv
- ID 순서 assert 검증

### 9. 피처 중요도
- Top 30 시각화 → output/feature_importance_phase6.png
- 도메인 피처가 Top 30에 몇 개 진입했는지 별도 출력

## 규칙
- 시각화 한글 폰트 적용
- 모든 MAE print 출력
- 결과를 claude_results.md에 Phase 6 섹션 추가
- run_phase6.py 작성만 하고 실행하지 말 것
- 커밋 메시지: "feat: Phase 6 - domain knowledge features + Optuna retuning"
- 커밋 + 푸시