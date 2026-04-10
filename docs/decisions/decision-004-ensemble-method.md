# Decision 004: Ensemble Method — Nelder-Mead

## 배경
- Phase 2: grid search 가중치
- Phase 8: Ridge + LGB stacking (Level 2)
- Phase 9: LGB stacking

## 결정
Nelder-Mead weighted averaging (normalized).

## 근거
- Stacking (Level 2 LGB): 추가 과적합 위험, 복잡도 증가
- Ridge stacking: 안전하지만 Nelder-Mead 대비 우위 없음
- Nelder-Mead: MAE 직접 최적화, 음수 weight 허용, 구현 간단

## 구현
```python
def objective(w, oof_m, y):
    w = w / (w.sum() + 1e-12)  # 내부 normalize
    return np.abs((oof_m * w).sum(axis=1) - y).mean()
```

## 주의
- weight normalization 필수 (objective 내부 + 최종)
- `result.fun` 대신 직접 계산 (normalization 미반영 문제)
- maxiter=3000 이상 권장

## 교훈
- 단순한 방법이 가장 안정적
- Level 2 stacking은 Phase 8~9에서 시도했으나 Nelder-Mead 대비 일관된 우위 없음
