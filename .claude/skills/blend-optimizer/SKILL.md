# Blend Optimizer

앙상블 가중치 최적화 패턴.

## Nelder-Mead (표준)
```python
from scipy.optimize import minimize

def objective(w, oof_m, y):
    w = w / (w.sum() + 1e-12)  # 내부 normalize
    return np.abs((oof_m * w).sum(axis=1) - y).mean()

x0 = np.ones(n_models) / n_models
result = minimize(objective, x0, args=(oof_matrix, y_train),
                  method='Nelder-Mead',
                  options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})
best_weights = result.x / (result.x.sum() + 1e-12)  # 최종 normalize
```

### 핵심
- objective 내부 + 최종 결과 모두 normalize (음수 weight 허용)
- `ensemble_cv = np.abs((oof_matrix * best_weights).sum(axis=1) - y).mean()`
  - `result.fun`이 아닌 직접 계산 (normalization 반영)

## Cross-Phase Blend
여러 Phase의 OOF를 결합:
```python
# P13s1 + P15 + P16 + P17 = 최대 30+ models
all_oofs = []
for phase, models in [(13, p13_oofs), (15, p15_oofs), ...]:
    for name, oof in models.items():
        all_oofs.append(oof)
cross_oof_matrix = np.column_stack(all_oofs)
```

## Multi-Seed Averaging
```python
# Seed 42, 123, 777 각각 학습 후 평균
for seed in [42, 123, 777]:
    # 학습 ...
    oof_seeds[seed] = oof
oof_avg = np.mean([oof_seeds[s] for s in seeds], axis=0)
```

## 주의
- Cross-phase blend: 모든 Phase가 같은 CV split을 사용해야 유효
- Phase 간 CV split이 다르면 blend가 과적합될 수 있음
