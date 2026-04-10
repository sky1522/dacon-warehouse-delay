# Phase 20: Clean Data + Distribution-Aware Modeling

## 가설
지금까지 모든 Phase는 오염된 base 위에 쌓였다:
- fillna(0)으로 order_inflow/robot_active/battery_mean 왜곡
- Train/Test distribution shift 무시
- MLP loss가 MAE (tail 무시)

깨끗한 전처리 + 분포 인식 기법 적용으로 -0.05~0.08 가능.

## 구현 계획

### Step 1: fillna(median) for 핵심 3개 피처
```python
MEDIAN_FILL = ['order_inflow_15m', 'robot_active', 'battery_mean']
for col in MEDIAN_FILL:
    med = combined.loc[combined['_is_train'] == 1, col].median()
    combined[col] = combined[col].fillna(med)
```
- train 기준 median 사용 (test leakage 방지)
- 이 3개만 변경, 나머지는 기존 방식 유지

### Step 2: Adversarial validation sample weight
```python
# train=0, test=1 분류기 학습
adv_clf = lgb.LGBMClassifier(n_estimators=200)
# OOF proba → sample_weight에 반영
# 0.5~1.5 범위로 clipping (과도한 가중치 방지)
```

### Step 3: MLP loss 변경
```python
# MAE → (RMSE + MAE) / 2
def custom_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    return (mae + rmse) / 2
```

### Step 4: Multi-seed (42, 2024, 777)
- 전체 파이프라인을 3 seed로 반복
- OOF 평균 → Nelder-Mead blend

### Step 5: Optuna HPO
- LGB huber: delta 재탐색
- MLP: learning_rate, dropout, architecture

## 예상 효과
- 보수적: 9.86 → 9.83 (4위)
- 중립: 9.86 → 9.80 (2위)
- 낙관: 9.86 → 9.77 (1위)

## 의존성
- Seed 777 학습 완료 대기
- 전처리 변경 → 체크포인트 전부 재학습 필요
- GPU 예산: ~6시간 (Colab T4)

## 리스크
- 전처리 변경으로 기존 체크포인트 무효화 (pipeline_version 변경으로 자동 감지)
- fillna(median)이 fillna(0) 대비 악화될 가능성 (일부 피처는 0이 맞음)
- Adversarial weight 과도 시 오히려 악화

## 검증 계획
1. fillna(median) 단독 효과 측정 (Step 1만)
2. Step 1 + Step 3 (MLP loss) 효과
3. 전체 파이프라인 (Step 1~5) CV 비교
4. Public 제출로 최종 확인
