# 데이콘 스마트 창고 출고 지연 예측 — Phase 5: Kaggle 우승 전략 적용

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- 최고 Public MAE: 10.203 (Phase 3B)
- 최고 CV MAE: 8.786 (LGB 80% + Cat 20% 앙상블)
- 피처: 194개
- Optuna best params (LGB): learning_rate=0.0129, num_leaves=185, max_depth=9, min_child_samples=80, reg_alpha=0.0574, reg_lambda=0.0042, feature_fraction=0.6005, bagging_fraction=0.7663, bagging_freq=1
- 참고 코드: run_phase3b.py

## ⚠️ 필수 사항
- concat+sort 후 test 분리 시 원본 ID 순서 복원 필수
- assert (submission['ID'] == sample_sub['ID']).all()
- run_phase5.py 파일 작성만 할 것. 실행하지 말 것.
- 스크립트가 길어질 수 있으므로 중간 진행상황 print 충분히 포함

## 핵심 전략 (Kaggle 우승자 노하우 기반)

### 전략 A: 대량 GroupBy 피처 생성
Kaggle Grandmaster Chris Deotte의 1위 전략: groupby(COL1)[COL2].agg(STAT)을 대량으로 생성하고 중요도 기반으로 선별.

### 전략 B: 타겟 Lag (재귀 예측)
같은 시나리오 내 이전 슬롯의 실제 타겟값을 피처로 활용. train에서는 실제값, test에서는 재귀적으로 예측값 사용.

## 수행 작업 (run_phase5.py)

### 1. 데이터 준비
- run_phase3b.py의 데이터 로드 + 시계열 피처 + 인터랙션 피처 로직 재사용
- layout_info 조인 포함
- implicit_timeslot 생성

### 2. 대량 GroupBy 피처 생성
아래 조합으로 대량 피처 자동 생성:

**그룹 키 (COL1):**
- layout_id
- layout_type (layout_info에서 조인)
- implicit_timeslot
- day_of_week

**집계 대상 (COL2) — 핵심 피처 15개:**
- order_inflow_15m, unique_sku_15m, robot_active, robot_idle, robot_charging
- battery_mean, low_battery_ratio, congestion_score, max_zone_density, pack_utilization
- robot_utilization, loading_dock_util, avg_trip_distance, charge_queue_length, fault_count_15m

**통계 (STAT):**
- mean, std, max, min, median

**생성 방법:**
```python
for col1 in group_keys:
    for col2 in agg_targets:
        for stat in stats:
            new_col = f'{col2}_by_{col1}_{stat}'
            mapping = train+test 합친 데이터에서 groupby(col1)[col2].agg(stat)
            # train과 test 모두에 매핑
```

**주의:** 
- 타겟(avg_delay_minutes_next_30m)을 COL2로 사용하면 안됨 (test에 없으므로)
- layout_id 기반 집계는 train+test 합쳐서 계산 (test의 unseen layout도 자체 통계 가짐)
- 생성된 피처 수와 이름 출력

### 3. 타겟 Lag 피처 (train만)
- target_lag1: 같은 시나리오에서 1슬롯 이전의 실제 타겟값
- target_lag2: 2슬롯 이전
- target_lag3: 3슬롯 이전
- target_roll3_mean: 이전 3슬롯의 타겟 이동평균
- target_cummax: 시나리오 시작부터 타겟 누적 최대값
- target_cumm_mean: 시나리오 시작부터 타겟 누적 평균

**첫 슬롯의 lag는 NaN → LightGBM이 자체 처리**

### 4. 피처 선별 — 중요도 기반
- 전체 피처(기존 194 + groupby + target lag)로 LightGBM 1회 학습 (3-Fold, 빠르게)
- 피처 중요도 상위 300개만 선택
- 선택된 피처 수와 선택/제거 피처 출력

### 5. 모델 A: 일반 LightGBM (Optuna params, 선별 피처)
- scenario_id GroupKFold 5-Fold
- log1p target
- CV MAE 출력
- OOF 예측 저장

### 6. 모델 B: 타겟 Lag 포함 LightGBM (재귀 예측)
**Train 시:**
- target_lag1~3, target_roll3_mean, target_cummax, target_cummean을 피처에 포함
- GroupKFold 5-Fold
- CV MAE 출력 (이 점수는 "실제 타겟값 사용" 기준이므로 낙관적)

**Test 재귀 예측:**
- 각 시나리오별로 슬롯 0부터 순차 예측
- 슬롯 0: target_lag는 NaN → 예측
- 슬롯 1: 슬롯 0의 예측값을 target_lag1에 넣고 → 예측
- 슬롯 2: 슬롯 1의 예측값을 target_lag1에, 슬롯 0을 target_lag2에 → 예측
- ... 반복
- 이 과정을 print로 진행상황 출력 (시나리오 500개마다)
- 재귀 예측 시 예측값 clip(0, None) 적용

### 7. 앙상블
- 모델 A OOF + 모델 B OOF + CatBoost OOF (모델 A와 같은 피처)
- 최적 가중치 탐색
- CV MAE 비교

### 8. 결과 테이블
=== Phase 5 결과 ===
생성된 GroupBy 피처: N개
선별 후 피처 수: N개
모델 A (GroupBy 피처): CV MAE X.XXXX
모델 B (타겟 Lag 포함): CV MAE X.XXXX (train 실제값 기준, 낙관적)
CatBoost: CV MAE X.XXXX
최종 앙상블: CV MAE X.XXXX
Phase 3B 대비 개선: X.XXXX

### 9. 제출 파일
- output/submission_phase5.csv
- 모델 B의 재귀 예측이 포함된 앙상블 사용
- ID 순서 assert 검증

## 규칙
- 시각화 한글 폰트 적용
- 모든 MAE print 출력
- 중간 진행상황 print 충분히 포함 (피처 생성 진행률, 학습 진행 등)
- 결과를 claude_results.md에 Phase 5 섹션 추가
- run_phase5.py 작성만 하고 실행하지 말 것
- 커밋 메시지: "feat: Phase 5 - mass groupby features + target lag"
- 커밋 + 푸시