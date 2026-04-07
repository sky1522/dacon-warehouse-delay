# 데이콘 스마트 창고 출고 지연 예측 — Phase 10: NN 추가 (TabNet + Keras MLP) + 8모델 스태킹

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- 최고 Public MAE: 10.097 (blend P7+P8, 31위)
- Phase 8 Public: 10.100, Phase 9 Public: 10.113 (과적합)
- 피처: 319개 (Phase 8과 동일)
- 10위 진입 목표: Public 9.99 이하
- 트리 모델만 6개 사용 중 → NN 추가로 다양성 폭증 노림

## ⚠️ 필수 사항
- run_phase8.py의 데이터 준비 + 피처 생성 로직 그대로 재사용 (319개 피처)
- concat+sort 후 test ID 순서 복원 필수
- assert (submission['ID'] == sample_sub['ID']).all()
- run_phase10.py 파일 작성만 할 것. 실행하지 말 것.
- float32 메모리 최적화
- 중간 진행상황 print 충분히 포함
- NN 학습은 시간 오래 걸리므로 중간 로그 자주 출력

## 전략: 트리 6모델 + NN 2모델 = 8모델 스태킹

### 1. 데이터 준비
- run_phase8.py와 동일하게 319개 피처 생성 (전체 함수 재사용)
- 샘플 가중치도 동일하게 생성

### 2. NN 전용 전처리 (트리 모델은 기존 데이터 그대로 사용)
- X_train_nn, X_test_nn 별도 생성
- StandardScaler로 정규화 (scaler는 train에 fit, test에 transform)
- NaN → 0.0으로 대체 (np.nan_to_num)
- inf → 0.0으로 대체
- target은 log1p 변환

### 3. 트리 6모델 학습 (Phase 8과 동일)
- 모델 1: LightGBM raw+MAE (Optuna params)
- 모델 2: LightGBM log1p+Huber
- 모델 3: LightGBM sqrt+MAE
- 모델 4: XGBoost raw+MAE
- 모델 5: CatBoost log1p+MAE
- 모델 6: CatBoost raw+MAE
- scenario_id GroupKFold 5-Fold
- OOF + test 예측 저장

### 4. NN 모델 1: Keras MLP

**구조:**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks

def build_mlp(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss='mae', metrics=['mae'])
    return model
```

**학습:**
- target: log1p
- batch_size=512
- epochs=100, early_stopping (patience=10, monitor='val_loss')
- ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-5)
- scenario_id GroupKFold 5-Fold
- Fold마다 model 새로 빌드
- OOF/test 예측은 expm1로 역변환
- 음수 예측은 0으로 클리핑

### 5. NN 모델 2: TabNet

**TabNet 설치 및 사용:**
```python
# pytorch-tabnet 사용
# !pip install pytorch-tabnet -q
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

model = TabNetRegressor(
    n_d=32, n_a=32,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    seed=42,
    verbose=10,
)
```

**학습:**
- target: log1p (reshape (-1, 1) 필요)
- max_epochs=100, patience=15
- batch_size=2048, virtual_batch_size=256
- eval_metric=['mae']
- scenario_id GroupKFold 5-Fold
- OOF/test 예측은 expm1로 역변환
- 음수 클리핑

### 6. Level 1 가중 평균 앙상블
- 8개 OOF (트리 6 + NN 2)로 scipy.optimize 가중치 최적화
- 트리만 가중평균 / 트리+NN 가중평균 두 개 비교
- 각각 CV MAE 출력

### 7. Level 2 LGB 스태킹
**메타 피처 (14개):**
- 8개 모델 OOF
- implicit_timeslot
- 핵심 5개 원본: order_inflow_15m, battery_mean, robot_active, pack_utilization, congestion_score

**메타 모델: LightGBM (작은 모델)**
- num_leaves=16, max_depth=4, n_estimators=200
- learning_rate=0.05
- np.nan_to_num으로 NaN 처리
- scenario_id GroupKFold 5-Fold

### 8. 결과 비교 출력
=== Phase 10 결과 ===
신규: NN 2개 추가 (Keras MLP + TabNet)
트리 모델 (Phase 8 동일):
lgb_raw_mae         : X.XXXX
lgb_log1p_huber     : X.XXXX
lgb_sqrt_mae        : X.XXXX
xgb_raw_mae         : X.XXXX
cat_log1p_mae       : X.XXXX
cat_raw_mae         : X.XXXX
NN 모델 (신규):
keras_mlp           : X.XXXX
tabnet              : X.XXXX
Level 1 가중 평균 (트리만):       X.XXXX
Level 1 가중 평균 (트리+NN):      X.XXXX
Level 2 LGB 스태킹 (8모델):       X.XXXX
최종 선택:                        X.XXXX
Phase 8 대비 개선:                X.XXXX

### 9. 제출 파일
- 최고 CV 모델로 output/submission_phase10.csv 생성
- ID 순서 assert 검증

### 10. 피처 중요도
- LGB raw+MAE 기준 Top 30 시각화
- output/feature_importance_phase10.png

## 규칙
- 시각화: 영문 폰트 사용 (Colab 환경, 한글 폰트 없음)
- 모든 MAE print 출력
- 결과를 claude_results.md에 Phase 10 섹션 추가
- run_phase10.py 작성만 하고 실행하지 말 것
- 커밋 메시지: "feat: Phase 10 - add NN models (Keras MLP + TabNet) for diversity"
- 커밋 + 푸시