# 데이콘 스마트 창고 출고 지연 예측 — Phase 12a: Queueing Theory 기반 피처 추가

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- 최고 Public MAE: 10.037 (Blend P10 70% + P7P8 30%, 20위)
- Phase 10 단독 Public: 10.039 (CV 8.577)
- Phase 11/11b 실패 확인 (adv weight가 일반화 망침)
- 목표: 10위 이내 (Public 9.94 이하)

## ⚠️ 필수 사항
- run_phase10.py를 기반으로 확장 (adv weight 절대 사용 금지)
- run_phase11b.py의 Drive 체크포인트 헬퍼(save_ckpt/load_ckpt)를 재사용
- 체크포인트 파일명: ckpt_phase12a_{model_name}.pkl
- submission 파일명: submission_phase12a.csv
- concat+sort 후 test ID 순서 복원 필수
- assert (submission['ID'] == sample_sub['ID']).all()
- run_phase12a.py 작성만 할 것. 실행하지 말 것.
- float32 메모리 최적화
- 중간 진행상황 print 충분히 포함
- drive.mount()는 스크립트 내부에서 호출 금지 (노트북에서 이미 마운트됨)

## 확인된 원본 컬럼
### train.csv (동적)
- robot_active, robot_idle, robot_charging, robot_utilization
- avg_trip_distance, avg_charge_wait
- charge_queue_length, pack_utilization, loading_dock_util
- order_inflow_15m, congestion_score, unique_sku_15m
- avg_package_weight_kg

### layout_info.csv (Phase 10에서 이미 join됨)
- robot_total, charger_count, pack_station_count
- aisle_width_avg, intersection_count, one_way_ratio
- floor_area_sqm 등

⚠️ dock_count 컬럼은 없음. loading_dock_util만 사용.
⚠️ avg_trip_time 컬럼 없음. avg_trip_distance로 대체.

## 전략
Phase 10과 완전히 동일한 구조 + Queueing Theory 피처 42개 추가만.
논문 3편의 이론 기반 피처가 핵심:
1. Unlocking Real-Time Decision-Making in Warehouses (Transport Research Part E, 2024)
2. Amazon Science: Congestion Prediction for Large Fleets of Mobile Robots (IEEE 2023)
3. Analysis of Robotic Mobile Fulfillment Systems Considering Congestion (MDPI 2021)

Phase 11의 adv weight / Transformer 는 절대 사용하지 않음.
8모델 (트리 6 + Keras MLP + TabNet) 그대로.

## 수행 작업 (run_phase12a.py)

### 1. 데이터 준비 + 피처 생성 (Phase 10과 동일)
run_phase10.py의 피처 생성 함수 전체 재사용:
- 시계열 피처 64개
- Phase 3B 인터랙션 8개
- Onset 8개
- Expanding mean 30개
- 비선형 임계값 7개
- 위상 6개
- 경쟁자 피처 54개
- Expanding 확장 20개
(총 319개)

### 2. ★ 신규: Queueing Theory 피처 추가 (42개)

피처 추가는 layout join이 완료된 df에 적용.
반드시 scenario_id, time_idx로 sort한 상태에서 diff/rolling 계산.

#### A) Utilization / Traffic Intensity (16개)
```python
EPS = 1e-3

# 기본 rho 4개
df['q_rho_robot'] = (df['robot_active'] / (df['robot_total'] + EPS)).clip(0, 0.99)
df['q_rho_charger'] = (df['robot_charging'] / (df['charger_count'] + EPS)).clip(0, 0.99)
df['q_rho_pack'] = df['pack_utilization'].clip(0, 0.99).astype('float32')
df['q_rho_loading'] = df['loading_dock_util'].clip(0, 0.99).astype('float32')

# 비선형 변환 (Pollaczek-Khinchin 공식 기반)
for name in ['robot', 'charger', 'pack', 'loading']:
    rho = df[f'q_rho_{name}']
    df[f'q_rho_{name}_sq'] = (rho ** 2).astype('float32')
    df[f'q_rho_{name}_inv'] = (1.0 / (1.0 - rho + EPS)).astype('float32')  # 1/(1-ρ)
    df[f'q_pk_{name}'] = (rho**2 / (1.0 - rho + EPS)).astype('float32')  # ρ²/(1-ρ) P-K 핵심
# 4개 기본 + 4×3 = 16개
```

#### B) Little's Law 기반 예상 대기시간 (6개)
```python
# W = L/λ
arrival_rate = (df['order_inflow_15m'] / 15.0).astype('float32')  # per minute

df['q_arrival_rate'] = arrival_rate
df['q_expected_charge_wait'] = (df['charge_queue_length'] / (arrival_rate + EPS)).astype('float32')

# 효과적 서비스율
df['q_effective_service_robot'] = (df['robot_active'] / (df['avg_trip_distance'] + EPS)).astype('float32')

# Arrival vs Service gap (양수면 대기열 증가 중)
df['q_arrival_service_gap'] = (arrival_rate - df['q_effective_service_robot']).astype('float32')

# Throughput
df['q_throughput_robot'] = (df['robot_active'] * (1.0 - df['congestion_score'])).astype('float32')
df['q_throughput_pack'] = (df['pack_station_count'] * df['pack_utilization']).astype('float32')
```

#### C) Bottleneck Detection (8개)
```python
# 각 단계별 사용률 (이미 위에서 계산)
stages_cols = ['q_rho_robot', 'q_rho_charger', 'q_rho_pack', 'q_rho_loading']
stages_df = df[stages_cols]

df['q_bottleneck_max'] = stages_df.max(axis=1).astype('float32')
df['q_bottleneck_min'] = stages_df.min(axis=1).astype('float32')
df['q_bottleneck_mean'] = stages_df.mean(axis=1).astype('float32')
df['q_bottleneck_std'] = stages_df.std(axis=1).astype('float32')
df['q_bottleneck_gap'] = (df['q_bottleneck_max'] - df['q_bottleneck_mean']).astype('float32')

# 연쇄 병목
df['q_cascade_load_pack'] = (df['q_rho_loading'] * df['q_rho_pack']).astype('float32')
df['q_cascade_pack_robot'] = (df['q_rho_pack'] * df['q_rho_robot']).astype('float32')
df['q_cascade_all'] = (df['q_rho_loading'] * df['q_rho_pack'] * df['q_rho_robot']).astype('float32')
```

#### D) Queue Stability Indicators (4개)
```python
df['q_unstable_robot'] = (df['q_rho_robot'] > 0.9).astype('float32')
df['q_unstable_charger'] = (df['q_rho_charger'] > 0.9).astype('float32')
df['q_unstable_pack'] = (df['q_rho_pack'] > 0.9).astype('float32')
df['q_unstable_count'] = (df['q_unstable_robot'] + df['q_unstable_charger'] + df['q_unstable_pack']).astype('float32')
```

#### E) Demand Surge / Time-varying (8개)
```python
# 반드시 scenario_id, time_idx로 sort된 상태에서
df = df.sort_values(['scenario_id', 'time_idx']).reset_index(drop=True)

# ρ 급변화
df['q_rho_robot_change'] = df.groupby('scenario_id')['q_rho_robot'].diff().fillna(0).astype('float32')
df['q_rho_robot_accel'] = df.groupby('scenario_id')['q_rho_robot_change'].diff().fillna(0).astype('float32')

# 최근 3 timestep의 평균 ρ
df['q_rho_robot_roll3'] = (
    df.groupby('scenario_id')['q_rho_robot']
      .rolling(3, min_periods=1).mean()
      .reset_index(level=0, drop=True)
      .astype('float32')
)

# Queue buildup rate
df['q_queue_growth'] = df.groupby('scenario_id')['charge_queue_length'].diff().fillna(0).astype('float32')
df['q_queue_growth_roll3'] = (
    df.groupby('scenario_id')['q_queue_growth']
      .rolling(3, min_periods=1).mean()
      .reset_index(level=0, drop=True)
      .astype('float32')
)

# Inflow 급변
df['q_inflow_change'] = df.groupby('scenario_id')['order_inflow_15m'].diff().fillna(0).astype('float32')
df['q_inflow_accel'] = df.groupby('scenario_id')['q_inflow_change'].diff().fillna(0).astype('float32')

# 누적 불안정 카운트 (시나리오 내)
df['q_cum_unstable'] = df.groupby('scenario_id')['q_unstable_count'].cumsum().astype('float32')

# 총: 16 + 6 + 8 + 4 + 8 = 42개
```

### 3. 전처리
- inf/NaN 처리: np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
- 모든 신규 피처 float32 확인

### 4. 피처 수 확인
```python
print(f"Phase 10 기존 피처: 319개")
print(f"Queueing theory 신규: 42개")
print(f"총 피처 수: ~361개")
```

### 5. 모델 학습 (Phase 10과 완전 동일)
- 트리 6모델 (기존 sample_weight만, adv weight 없음)
- Keras MLP
- TabNet
- 체크포인트: ckpt_phase12a_{model_name}.pkl
- Drive 동시 저장

### 6. Level 1 가중 평균
- 트리만
- 트리+NN 8모델
- 두 가지 비교

### 7. Level 2 LGB 스태킹 (8모델)
- 메타 피처: 8 OOF + 6 원본 = 14개
- np.nan_to_num 적용

### 8. 결과 출력
=== Phase 12a 결과 ===
신규 피처: Queueing Theory 42개
총 피처 수: ~361개
트리 6모델 CV MAE:
...
NN 모델:
keras_mlp           : X.XXXX
tabnet              : X.XXXX
Level 1 트리만:           X.XXXX
Level 1 트리+NN (8모델):  X.XXXX
Level 2 LGB 스태킹:       X.XXXX
최종 선택:                X.XXXX
Phase 10 대비 개선:       X.XXXX
Queueing theory 피처 중 Top 30 진입: X개
(q_ 접두사 필터링)

### 9. 제출 파일
- output/submission_phase12a.csv
- /content/drive/MyDrive/dacon_ckpt/submission_phase12a.csv 동시 저장

### 10. 피처 중요도
- LGB raw+MAE 기준 Top 30 시각화
- feature_importance_phase12a.png
- Queueing theory 피처(q_ 접두사) Top 30 진입 개수 별도 출력

## 규칙
- 시각화: 영문 폰트
- 결과를 claude_results.md에 Phase 12a 섹션 추가
- run_phase12a.py 작성만, 실행 금지
- 커밋 메시지: "feat: Phase 12a - queueing theory features (Little's Law, Pollaczek-Khinchin, bottleneck detection)"
- 커밋 + 푸시