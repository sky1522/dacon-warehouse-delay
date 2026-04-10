"""
Phase 20 Pre-EDA: 4가지 질문에 답변
Q1: Adversarial AUC (train vs test shift)
Q2: Median vs Zero fill 차이
Q3: Adversarial holdout 검증
Q4: Phase 16 OOF holdout MAE
"""

import pandas as pd
import numpy as np
import pickle
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.optimize import minimize

os.makedirs('output/phase20_eda', exist_ok=True)

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')

N_TRAIN = len(train)
y_train = train['avg_delay_minutes_next_30m'].values

# Minimal feature set (raw numeric only, no FE)
train_m = train.merge(layout, on='layout_id', how='left')
test_m = test.merge(layout, on='layout_id', how='left')

drop_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m']
num_cols = [c for c in train_m.columns
            if c not in drop_cols and train_m[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

X_train_raw = train_m[num_cols].fillna(0).astype('float32')
X_test_raw = test_m[num_cols].fillna(0).astype('float32')

print(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}, Features: {len(num_cols)}")


# ============================================================
# Q1: Adversarial AUC
# ============================================================
print("\n" + "=" * 60)
print("=== Q1: Adversarial Validation AUC ===")
print("=" * 60)

adv_X = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
adv_y = np.concatenate([np.zeros(N_TRAIN), np.ones(len(X_test_raw))])

clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbosity=-1,
    random_state=42,
)

# 5-fold OOF
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
adv_oof = np.zeros(len(adv_y))

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(adv_X, adv_y)):
    clf.fit(
        adv_X.iloc[tr_idx], adv_y[tr_idx],
        eval_set=[(adv_X.iloc[va_idx], adv_y[va_idx])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
    )
    adv_oof[va_idx] = clf.predict_proba(adv_X.iloc[va_idx])[:, 1]

adv_auc = roc_auc_score(adv_y, adv_oof)
print(f"\nAdversarial AUC: {adv_auc:.4f}")

if adv_auc > 0.7:
    print("  -> Shift SEVERE: adversarial weight will be effective")
elif adv_auc > 0.55:
    print("  -> Shift MODERATE: adversarial weight may help marginally")
else:
    print("  -> Shift MINIMAL: adversarial weight unnecessary")

# Train samples의 adv_proba
train_adv_proba = adv_oof[:N_TRAIN]
print(f"\nTrain adv_proba stats: mean={train_adv_proba.mean():.4f}, "
      f"std={train_adv_proba.std():.4f}, "
      f"min={train_adv_proba.min():.4f}, max={train_adv_proba.max():.4f}")

# Top feature importance for adversarial
imp = pd.DataFrame({'feature': num_cols, 'importance': clf.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
print(f"\nTop 10 adversarial features:")
print(imp.head(10).to_string(index=False))


# ============================================================
# Q2: Median vs Zero Fill
# ============================================================
print("\n" + "=" * 60)
print("=== Q2: Median vs Zero Fill ===")
print("=" * 60)

FILL_COLS = ['order_inflow_15m', 'robot_active', 'battery_mean']

for col in FILL_COLS:
    nan_rate = train[col].isna().mean()
    median_val = train[col].median()
    mean_val = train[col].mean()

    # fill(0) distribution
    filled_0 = train[col].fillna(0)
    # fill(median) distribution
    filled_med = train[col].fillna(median_val)

    print(f"\n  {col}:")
    print(f"    NaN rate: {nan_rate*100:.1f}%")
    print(f"    Median: {median_val:.2f}, Mean: {mean_val:.2f}")
    print(f"    fill(0) mean: {filled_0.mean():.2f}, std: {filled_0.std():.2f}")
    print(f"    fill(median) mean: {filled_med.mean():.2f}, std: {filled_med.std():.2f}")
    print(f"    Difference: mean shift {abs(filled_0.mean() - filled_med.mean()):.2f}")

    # NaN rows의 target 평균 vs non-NaN rows
    nan_mask = train[col].isna()
    if nan_mask.any():
        y_nan = y_train[nan_mask]
        y_nonan = y_train[~nan_mask]
        print(f"    Target (NaN rows): mean={y_nan.mean():.2f}")
        print(f"    Target (non-NaN):  mean={y_nonan.mean():.2f}")


# ============================================================
# Q3: Adversarial Holdout
# ============================================================
print("\n" + "=" * 60)
print("=== Q3: Adversarial Holdout Validation ===")
print("=" * 60)

# Top 20% test-like train samples
threshold_20 = np.percentile(train_adv_proba, 80)
holdout_mask = train_adv_proba >= threshold_20
remain_mask = ~holdout_mask

n_holdout = holdout_mask.sum()
n_remain = remain_mask.sum()
print(f"Holdout (top 20% test-like): {n_holdout}")
print(f"Remain (train-like):         {n_remain}")

# Distribution comparison
KEY_COLS = ['order_inflow_15m', 'congestion_score', 'pack_utilization', 'robot_active', 'battery_mean']

print(f"\n{'Column':<25} {'Train_remain':>12} {'Holdout':>12} {'Test':>12} {'Hold-Test gap':>12}")
print("-" * 75)
for col in KEY_COLS:
    if col not in train.columns:
        continue
    tr_mean = train.loc[remain_mask, col].mean()
    ho_mean = train.loc[holdout_mask, col].mean()
    te_mean = test[col].mean() if col in test.columns else float('nan')
    gap = abs(ho_mean - te_mean) if not np.isnan(te_mean) else float('nan')
    print(f"{col:<25} {tr_mean:>12.2f} {ho_mean:>12.2f} {te_mean:>12.2f} {gap:>12.2f}")

# Holdout target distribution
y_holdout = y_train[holdout_mask]
y_remain = y_train[remain_mask]
print(f"\nTarget distribution:")
print(f"  Remain:  mean={y_remain.mean():.2f}, median={np.median(y_remain):.2f}, max={y_remain.max():.2f}")
print(f"  Holdout: mean={y_holdout.mean():.2f}, median={np.median(y_holdout):.2f}, max={y_holdout.max():.2f}")


# ============================================================
# Q4: Phase 16 OOF Holdout MAE
# ============================================================
print("\n" + "=" * 60)
print("=== Q4: Phase 16 OOF Holdout MAE ===")
print("=" * 60)

P16_MODELS = ['lgb_raw', 'lgb_huber', 'lgb_sqrt', 'xgb', 'cat_log1p', 'cat_raw', 'mlp']

p16_oofs = {}
for name in P16_MODELS:
    path = f'output/ckpt_phase16_{name}.pkl'
    if not os.path.exists(path):
        drive_path = f'/content/drive/MyDrive/dacon_ckpt/ckpt_phase16_{name}.pkl'
        if os.path.exists(drive_path):
            import shutil
            shutil.copy(drive_path, path)

    if not os.path.exists(path):
        print(f"  MISSING: {name}")
        continue

    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    if 'oof' not in ckpt or len(ckpt['oof']) != len(y_train):
        print(f"  BAD SHAPE: {name}")
        continue
    p16_oofs[name] = ckpt['oof']

if len(p16_oofs) == len(P16_MODELS):
    # Nelder-Mead normalized blend
    def nm_obj(w, oof_m, y):
        w = w / (w.sum() + 1e-12)
        return np.abs((oof_m * w).sum(axis=1) - y).mean()

    oof_mat = np.column_stack([p16_oofs[n] for n in P16_MODELS])
    x0 = np.ones(len(P16_MODELS)) / len(P16_MODELS)
    res = minimize(nm_obj, x0, args=(oof_mat, y_train),
                   method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})
    w = res.x / (res.x.sum() + 1e-12)
    p16_oof = (oof_mat * w).sum(axis=1)

    full_mae = np.abs(p16_oof - y_train).mean()
    remain_mae = np.abs(p16_oof[remain_mask] - y_train[remain_mask]).mean()
    holdout_mae = np.abs(p16_oof[holdout_mask] - y_train[holdout_mask]).mean()

    print(f"\n  Full CV MAE:      {full_mae:.4f}")
    print(f"  Train remain MAE: {remain_mae:.4f}")
    print(f"  Holdout MAE:      {holdout_mae:.4f}")
    print(f"  Public MAE:       9.87947")
    print(f"  Holdout-Public gap: {abs(holdout_mae - 9.87947):.4f}")

    if abs(holdout_mae - 9.87947) < 0.5:
        print(f"  -> Holdout is a GOOD proxy for Public")
    else:
        print(f"  -> Holdout is NOT a reliable proxy for Public")
else:
    print(f"\n  Only {len(p16_oofs)}/{len(P16_MODELS)} checkpoints available")
    print(f"  Run Phase 16 first or restore from Drive")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("=== SUMMARY ===")
print("=" * 60)
print(f"Q1 Adversarial AUC: {adv_auc:.4f}")
print(f"Q2 Median fill cols: {FILL_COLS}")
print(f"Q3 Holdout size: {n_holdout} ({n_holdout/N_TRAIN*100:.1f}%)")
print(f"Q4 Phase 16 checkpoints: {len(p16_oofs)}/{len(P16_MODELS)}")

# Save
results = {
    'adv_auc': adv_auc,
    'train_adv_proba': train_adv_proba,
    'holdout_mask': holdout_mask,
    'adv_feature_importance': imp,
}
with open('output/phase20_eda/eda_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to output/phase20_eda/")
