run_phase20_clean.py 5개 버그 수정. 실행 금지.

## Fix 1 (Critical): roc_auc_score import
```python
# 상단 import 추가
from sklearn.metrics import roc_auc_score
```

## Fix 2 (Critical): Holdout MAE 비편향 계산

ensemble weight optimization을 train_remain 20% 제외한 80%에서 수행:

```python
# 기존: full OOF에서 weight 최적화
# 수정: holdout 제외한 train_remain에서만 최적화

train_remain_mask = ~holdout_mask
def objective(w, oof_m, y, mask):
    w = w / (w.sum() + 1e-12)
    return np.abs((oof_m[mask] * w).sum(axis=1) - y[mask]).mean()

x0 = np.ones(len(model_names)) / len(model_names)
result = minimize(
    objective, x0, 
    args=(oof_matrix, y_train, train_remain_mask),
    method='Nelder-Mead',
    options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000}
)
best_weights = result.x / (result.x.sum() + 1e-12)

# 이제 holdout은 weight 결정에 영향 없음, unbiased 검증 가능
ensemble_oof = (oof_matrix * best_weights).sum(axis=1)
train_remain_mae = np.abs(ensemble_oof[train_remain_mask] - y_train[train_remain_mask]).mean()
holdout_mae = np.abs(ensemble_oof[holdout_mask] - y_train[holdout_mask]).mean()

print(f"Train remain MAE (weight optimized on this): {train_remain_mae:.4f}")
print(f"Holdout MAE (UNBIASED, Public 근사):          {holdout_mae:.4f}")
print(f"Phase 16 holdout baseline:                    9.7341")
print(f"Expected Public range:                        [{holdout_mae*1.01:.4f}, {holdout_mae*1.02:.4f}]")
```

## Fix 3 (Medium): Adversarial split layout-aware

```python
# 기존: StratifiedKFold
# 수정: GroupKFold by layout_id

from sklearn.model_selection import GroupKFold
import numpy as np

# 50 shared layouts는 train/test 양쪽에 있음
# → adversarial classifier가 layout_id로 cheat 가능
# GroupKFold로 같은 layout이 train/valid 양쪽에 없도록

adv_groups = np.concatenate([
    train_df['layout_id'].values,
    test_df['layout_id'].values
])

cv_adv = GroupKFold(n_splits=5)
adv_proba = np.zeros(len(adv_X), dtype='float32')

for fold_idx, (tr_idx, va_idx) in enumerate(cv_adv.split(adv_X, adv_y, groups=adv_groups)):
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        max_depth=7, min_child_samples=100,
        verbosity=-1, random_state=42
    )
    clf.fit(adv_X.iloc[tr_idx], adv_y[tr_idx])
    adv_proba[va_idx] = clf.predict_proba(adv_X.iloc[va_idx])[:, 1].astype('float32')

adv_auc = roc_auc_score(adv_y, adv_proba)
print(f"Adversarial AUC (group-aware): {adv_auc:.4f}")
print(f"  Note: may be lower than 0.989 due to group split")
```

## Fix 4 (Medium): Fold-local median fill

fillna를 CV fold loop 안으로 이동. 각 fold의 train 80%에서만 median 계산:

```python
# 기존: 전역 median fill
# 수정: fold 내부에서 train fold의 median으로 fill

# median fill 제거 → CV loop 안에서 처리
MEDIAN_FILL_COLS = [...]  # 30개 컬럼

# 학습 루프 내부:
for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr = X_train.iloc[tr_idx].copy()
    X_va = X_train.iloc[va_idx].copy()
    X_te = X_test.copy()
    
    # Fold-local median fill
    for col in MEDIAN_FILL_COLS:
        if col in X_tr.columns:
            med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(med)
            X_va[col] = X_va[col].fillna(med)
            X_te[col] = X_te[col].fillna(med)
    
    # 나머지는 0
    X_tr = X_tr.fillna(0)
    X_va = X_va.fillna(0)
    X_te = X_te.fillna(0)
    
    # ... 학습 진행
```

⚠️ 이건 구조가 크게 바뀜. **대안**: 간단하게 그대로 두고 "mild leakage" 인지만 하고 진행.
   → Fix 4는 **Phase 20.1에서 하고 지금은 스킵** 권장.

## Fix 5 (Critical): MLP/TabNet에 sample_weight 추가

```python
# MLP fit (line ~1103 근처)
mlp.fit(
    X_tr_scaled, y_tr_scaled,
    validation_data=(X_va_scaled, y_va_scaled),
    epochs=100,
    batch_size=512,
    sample_weight=final_weight[tr_idx],  # ← 추가!
    callbacks=[...],
    verbose=0,
)

# TabNet fit (line ~1126 근처)
tabnet.fit(
    X_tr_scaled, y_tr_scaled.reshape(-1, 1),
    eval_set=[(X_va_scaled, y_va_scaled.reshape(-1, 1))],
    weights=final_weight[tr_idx],  # ← 추가!
    max_epochs=200,
    patience=20,
    batch_size=1024,
)
```

⚠️ TabNet의 weights 파라미터는 라이브러리 버전에 따라 다름. 안 되면 skip.

## 우선순위 적용

- Fix 1: 반드시 (스크립트 실행 불가)
- Fix 2: 반드시 (검증 신뢰성)
- Fix 5: 반드시 (핵심 변경사항이 MLP에 적용 안 됨)
- Fix 3: 권장 (AUC 재측정, 여전히 높으면 효과 확실)
- Fix 4: 스킵 (Phase 20.1로 연기)

## 작업 후
- claude_results.md에 수정 내역 추가
- 커밋: "fix: Phase 20 - 4 bugs (roc_auc import, unbiased holdout, group-aware adv, MLP sample_weight)"
- 푸시