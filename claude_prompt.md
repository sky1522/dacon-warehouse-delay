# 데이콘 스마트 창고 출고 지연 예측 — Phase 9: Optuna 재튜닝 + Seed Blending

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- 최고 Public MAE: 10.100 (Phase 8, 30위)
- 최고 CV MAE: 8.653 (Phase 8 LGB 스태킹)
- 피처: 319개
- 10위 진입 목표: Public 9.99 이하

## ⚠️ 필수 사항
- run_phase8.py의 데이터 준비 + 피처 생성 로직을 그대로 재사용 (319개 피처)
- concat+sort 후 test ID 순서 복원 필수
- assert (submission['ID'] == sample_sub['ID']).all()
- run_phase9.py 파일 작성만 할 것. 실행하지 말 것.
- float32 메모리 최적화
- 중간 진행상황 print 포함

## 전략: Optuna 재튜닝 + Multi-Seed + 스태킹

### 1. 데이터 준비
- run_phase8.py와 동일하게 319개 피처 생성
- 샘플 가중치 동일

### 2. Optuna 재튜닝 (319개 피처 기준, 30 trials)
Phase 3B의 Optuna는 194개 피처 기준. 319개 피처에서 최적 파라미터가 다를 수 있음.

LightGBM raw+MAE 대상:
탐색 범위:
- num_leaves: 31~255
- max_depth: 4~12
- learning_rate: 0.01~0.1 (log uniform)
- n_estimators: 고정 2000, early_stopping_rounds=100
- min_child_samples: 10~100
- reg_alpha: 1e-3~10.0 (log uniform)
- reg_lambda: 1e-3~10.0 (log uniform)
- feature_fraction: 0.4~0.9
- bagging_fraction: 0.5~0.9
- bagging_freq: 1~7

목적함수: scenario_id GroupKFold 5-Fold MAE
sampler=optuna.samplers.TPESampler(seed=42)
매 10 trial마다 best MAE 출력

### 3. Multi-Seed 6모델 (각 seed 3개)
Optuna best params를 모델 1에 적용.
각 모델을 seed 42, 123, 777로 3번 학습 → OOF/test 예측을 seed별로 평균.
이렇게 하면 모델 안정성이 높아지고 CV-Public 갭이 줄어듦.

모델 구성 (Phase 8과 동일한 6모델, seed만 3개):

**모델 1: LightGBM raw+MAE (Optuna NEW params)**
- seeds: [42, 123, 777]
- target_transform='none', objective='mae'

**모델 2: LightGBM log1p+Huber**
- seeds: [42, 123, 777]
- Phase 8과 동일 params

**모델 3: LightGBM sqrt+MAE**
- seeds: [42, 123, 777]
- Phase 8과 동일 params

**모델 4: XGBoost raw+MAE**
- seeds: [42, 123, 777]
- Phase 8과 동일 params

**모델 5: CatBoost log1p+MAE**
- seeds: [42, 123, 777]
- Phase 8과 동일 params

**모델 6: CatBoost raw+MAE**
- seeds: [42, 123, 777]
- Phase 8과 동일 params

각 모델 seed별 학습 → OOF/test를 3-seed 평균 → 6개 OOF

### 4. Level 1 가중 평균 앙상블
- 6개 seed-averaged OOF로 가중치 최적화
- CV MAE 출력

### 5. Level 2 스태킹
메타 피처: 6개 OOF + implicit_timeslot + order_inflow_15m + battery_mean + robot_active + pack_utilization + congestion_score = 12개
- NaN이 있으면 np.nan_to_num(X, nan=0.0)으로 처리
- LightGBM 메타 모델 (num_leaves=16, max_depth=4, n_estimators=200)
- scenario_id GroupKFold 5-Fold

### 6. 최종 결과
최고 CV 모델로 output/submission_phase9.csv 생성

### 7. 결과 비교
=== Phase 9 결과 ===
Optuna 재튜닝: X.XXXX (기존 8.7494 → 신규 X.XXXX)
Optuna best params: {...}
모델별 CV MAE (seed-averaged):
LGB raw+MAE:        X.XXXX
LGB log1p+Huber:    X.XXXX
LGB sqrt+MAE:       X.XXXX
XGB raw+MAE:        X.XXXX
CatBoost log1p+MAE: X.XXXX
CatBoost raw+MAE:   X.XXXX
Level 1 가중 평균: X.XXXX
Level 2 LGB 스태킹: X.XXXX
Phase 8 대비 개선: X.XXXX

## 규칙
- 시각화: 영문 폰트 사용 (Colab 환경, 한글 폰트 없음)
- 모든 MAE print 출력
- 결과를 claude_results.md에 Phase 9 섹션 추가
- run_phase9.py 작성만 하고 실행하지 말 것
- 커밋 메시지: "feat: Phase 9 - Optuna retuning + seed blending + stacking"
- 커밋 + 푸시