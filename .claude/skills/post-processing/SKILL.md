# Post-processing

후처리 전략 — calibration, clipping, averaging.

## Isotonic Regression (Nested CV)

### 패턴
```python
from sklearn.isotonic import IsotonicRegression

oof_calibrated = np.zeros(len(y_train), dtype='float32')
for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso.fit(oof_pred[tr_idx], y_train[tr_idx])
    oof_calibrated[va_idx] = iso.predict(oof_pred[va_idx])

cv_before = np.abs(oof_pred - y_train).mean()
cv_after = np.abs(oof_calibrated - y_train).mean()
```

### 적용 판단
- 개선 > 0.001: 적용
- 개선 <= 0.001: skip (과적합 위험)
- 반드시 같은 folds 사용 (leakage 방지)

### Test 적용
```python
# Train 전체로 fit → test 적용
iso_final = IsotonicRegression(out_of_bounds='clip', increasing=True)
iso_final.fit(oof_pred, y_train)
test_calibrated = iso_final.predict(test_pred)
```

## Prediction Clip 전략

### 기본
```python
test_pred = np.clip(test_pred, 0, 500)  # 보수적
```

### 공격적 (Bin 9 공략 시)
```python
test_pred = np.clip(test_pred, 0, 1000)  # Phase 18 사용
```

### 판단 기준
- train max: 715.86
- prediction max 관찰 후 결정
- clip이 너무 낮으면 Bin 9 예측 못함

## Rank Averaging vs Value Averaging

### Value Averaging (기본)
```python
blend = w1 * pred1 + w2 * pred2  # 단위: 원래 값
```

### Rank Averaging (분포 보존)
```python
from scipy.stats import rankdata
rank1 = rankdata(pred1) / len(pred1)
rank2 = rankdata(pred2) / len(pred2)
blend_rank = w1 * rank1 + w2 * rank2
# → rank를 다시 value로 변환 (train target 분포 기준)
```

### 선택 기준
- 모델 간 예측 scale이 비슷: value averaging
- 모델 간 예측 scale이 다름 (log1p vs raw): rank averaging 검토
- 현재 프로젝트: value averaging (Nelder-Mead가 scale 차이 자동 보정)
