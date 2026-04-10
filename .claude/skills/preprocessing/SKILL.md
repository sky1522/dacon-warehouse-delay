# Preprocessing

데이터 전처리 전략 — NaN, outlier, distribution shift.

## NaN 전략

### fillna(0) 금지
- `order_inflow_15m`, `robot_active`, `battery_mean` 등 핵심 피처에 fillna(0) 사용 시 실제 값 0과 결측을 구분 불가
- 0이 valid value인 피처는 반드시 다른 전략 사용

### 권장 전략
```python
# 핵심 피처: median imputation
MEDIAN_FILL_COLS = ['order_inflow_15m', 'robot_active', 'battery_mean']
for col in MEDIAN_FILL_COLS:
    med = combined.loc[combined['_is_train'] == 1, col].median()
    combined[col] = combined[col].fillna(med)

# 파생 피처: 0 또는 np.nan_to_num
# lag/rolling/expanding: NaN은 자연스러움 (첫 timestep), 0 허용
```

### Missing indicator
```python
# 결측 여부 자체가 정보일 수 있음
combined[f'{col}_missing'] = combined[col].isnull().astype(int)
```

## IQR Outlier 분석

### 판단 기준
```python
Q1, Q3 = col.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
n_outliers = ((col < lower) | (col > upper)).sum()
```

### 제거 vs 유지
- **유지**: target 관련 outlier (높은 delay는 실제 현상)
- **유지**: 피처 outlier이지만 target 예측에 유용한 경우
- **제거/clip**: 명백한 데이터 오류 (음수 시간, 100% 초과 비율)

## Train/Test Distribution Shift

### 체크 방법
```python
for col in feature_cols:
    train_mean = X_train[col].mean()
    test_mean = X_test[col].mean()
    diff_pct = abs(train_mean - test_mean) / (abs(train_mean) + 1e-6) * 100
    if diff_pct > 30:
        print(f"WARNING: {col} shift {diff_pct:.1f}%")
```

### 대응
- 차이 >30% 피처: 정규화 또는 제거 검토
- Adversarial validation으로 test-like train samples 가중치 부여

## Adversarial Validation
```python
# train=0, test=1로 분류기 학습
adv_target = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
adv_X = pd.concat([X_train, X_test])
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
# OOF로 train sample별 "test-like" 확률 계산
# → sample_weight에 반영
```
