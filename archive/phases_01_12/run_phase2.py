import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from matplotlib.patches import Patch
import warnings, sys
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 데이터 준비 (Phase 1 로직 + 버그 수정)
# ============================================================
print("=== 데이터 로드 ===", flush=True)
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

# 원본 순서 보존을 위한 인덱스 컬럼 추가
train['_is_train'] = 1
test['_is_train'] = 0
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined['_original_idx'] = range(len(combined))  # 버그 수정: 원본 순서 보존

combined['implicit_timeslot'] = combined.groupby('scenario_id').cumcount()
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)

# --- 시계열 피처 ---
print("=== 시계열 피처 생성 ===", flush=True)
ts_cols = [
    'battery_mean', 'low_battery_ratio', 'order_inflow_15m', 'congestion_score',
    'robot_idle', 'robot_charging', 'max_zone_density', 'pack_utilization'
]

for col in ts_cols:
    for lag_val in [1, 2, 3]:
        combined[f'{col}_lag{lag_val}'] = combined.groupby('scenario_id')[col].shift(lag_val)

for col in ts_cols:
    l1, l2, l3 = combined[f'{col}_lag1'], combined[f'{col}_lag2'], combined[f'{col}_lag3']
    combined[f'{col}_roll3_mean'] = pd.concat([l1, l2, l3], axis=1).mean(axis=1)
    combined[f'{col}_roll3_std'] = pd.concat([l1, l2, l3], axis=1).std(axis=1)
    l4 = combined.groupby('scenario_id')[col].shift(4)
    l5 = combined.groupby('scenario_id')[col].shift(5)
    combined[f'{col}_roll5_mean'] = pd.concat([l1, l2, l3, l4, l5], axis=1).mean(axis=1)
    combined[f'{col}_roll5_std'] = pd.concat([l1, l2, l3, l4, l5], axis=1).std(axis=1)

for col in ['battery_mean', 'order_inflow_15m', 'congestion_score', 'robot_idle']:
    combined[f'{col}_diff1'] = combined[col] - combined[f'{col}_lag1']

for col in ['order_inflow_15m', 'fault_count_15m', 'near_collision_15m', 'blocked_path_15m']:
    combined[f'{col}_cumsum'] = combined.groupby('scenario_id')[col].cumsum()

print(f"시계열 피처 수: {len([c for c in combined.columns if any(s in c for s in ['_lag','_roll','_diff1','_cumsum'])])}", flush=True)

# --- 기존 피처 ---
combined = combined.merge(layout, on='layout_id', how='left')
layout_type_map = {t: i for i, t in enumerate(layout['layout_type'].unique())}
combined['layout_type_encoded'] = combined['layout_type'].map(layout_type_map).fillna(-1).astype(int)
combined['robot_per_area'] = combined['robot_total'] / combined['floor_area_sqm'].replace(0, np.nan)
combined['charger_per_robot'] = combined['charger_count'] / combined['robot_total'].replace(0, np.nan)
combined['packstation_per_robot'] = combined['pack_station_count'] / combined['robot_total'].replace(0, np.nan)
combined['order_per_active_robot'] = combined['order_inflow_15m'] / combined['robot_active'].replace(0, np.nan)
combined['sku_per_packstation'] = combined['unique_sku_15m'] / combined['pack_station_count'].replace(0, np.nan)
combined['battery_bottleneck'] = combined['low_battery_ratio'] * combined['charge_queue_length']
combined['battery_spread'] = combined['battery_mean'] - combined['battery_std']

train_part = combined[combined['_is_train'] == 1]
missing_counts = train_part.isnull().sum().sort_values(ascending=False)
top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith('_')]
for col in top10_missing:
    if col in combined.columns:
        combined[f'{col}_missing'] = combined[col].isnull().astype(int)

# 원본 순서 복원 후 분리
combined = combined.sort_values('_original_idx').reset_index(drop=True)
train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()

drop_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']
feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
print(f"총 피처 수: {len(feature_cols)}", flush=True)

X = train_fe[feature_cols]
y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y)
groups = train_fe['scenario_id']
X_test = test_fe[feature_cols]

# ID 순서 검증
assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID 순서 불일치!"
print("ID 순서 검증 통과!", flush=True)

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y_log, groups))

# ============================================================
# 2. 모델 A: LightGBM
# ============================================================
print("\n=== 모델 A: LightGBM 학습 ===", flush=True)
lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))
lgb_maes = []
lgb_models = []

for fold, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]

    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=2000, learning_rate=0.03,
        num_leaves=63, random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    lgb_oof[va_idx] = pred
    lgb_test += np.clip(np.expm1(m.predict(X_test)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    lgb_maes.append(mae)
    lgb_models.append(m)
    print(f"  LGB Fold {fold+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

lgb_cv = mean_absolute_error(y, lgb_oof)
print(f"  LightGBM CV MAE: {lgb_cv:.4f}", flush=True)

# ============================================================
# 2. 모델 B: CatBoost
# ============================================================
print("\n=== 모델 B: CatBoost 학습 ===", flush=True)
cat_oof = np.zeros(len(X))
cat_test = np.zeros(len(X_test))
cat_maes = []

for fold, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]

    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=6,
        loss_function='MAE', random_seed=42, verbose=0
    )
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100)

    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    cat_oof[va_idx] = pred
    cat_test += np.clip(np.expm1(m.predict(X_test)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    cat_maes.append(mae)
    print(f"  Cat Fold {fold+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

cat_cv = mean_absolute_error(y, cat_oof)
print(f"  CatBoost CV MAE: {cat_cv:.4f}", flush=True)

# ============================================================
# 2. 모델 C: XGBoost
# ============================================================
print("\n=== 모델 C: XGBoost 학습 ===", flush=True)
xgb_oof = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))
xgb_maes = []

for fold, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]

    m = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        eval_metric='mae', tree_method='hist', random_state=42,
        verbosity=0, early_stopping_rounds=100
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    xgb_oof[va_idx] = pred
    xgb_test += np.clip(np.expm1(m.predict(X_test)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    xgb_maes.append(mae)
    print(f"  XGB Fold {fold+1} MAE: {mae:.4f} (iter: {m.best_iteration})", flush=True)

xgb_cv = mean_absolute_error(y, xgb_oof)
print(f"  XGBoost CV MAE: {xgb_cv:.4f}", flush=True)

# ============================================================
# 3. 앙상블 가중치 최적화
# ============================================================
print("\n=== 앙상블 가중치 최적화 ===", flush=True)

def ensemble_mae(weights):
    w = weights / weights.sum()
    blended = w[0] * lgb_oof + w[1] * cat_oof + w[2] * xgb_oof
    return mean_absolute_error(y, blended)

result = minimize(
    ensemble_mae,
    x0=np.array([1/3, 1/3, 1/3]),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)

best_w = result.x / result.x.sum()
ensemble_oof = best_w[0] * lgb_oof + best_w[1] * cat_oof + best_w[2] * xgb_oof
ens_cv = mean_absolute_error(y, ensemble_oof)

print(f"최적 가중치 - LightGBM: {best_w[0]:.4f}, CatBoost: {best_w[1]:.4f}, XGBoost: {best_w[2]:.4f}", flush=True)
print(f"Ensemble CV MAE: {ens_cv:.4f}", flush=True)

# ============================================================
# 4. 결과 비교
# ============================================================
print("\n=== 모델별 CV MAE 비교 ===", flush=True)
print(f"LightGBM:  {lgb_cv:.4f}")
print(f"CatBoost:  {cat_cv:.4f}")
print(f"XGBoost:   {xgb_cv:.4f}")
print(f"Ensemble:  {ens_cv:.4f} (가중치: lgb={best_w[0]:.2f}, cat={best_w[1]:.2f}, xgb={best_w[2]:.2f})")
print(f"Phase 1:   8.8508")
print(f"베이스라인: 9.1820")

# ============================================================
# 5. 제출 파일 생성
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
test_blended = best_w[0] * lgb_test + best_w[1] * cat_test + best_w[2] * xgb_test
test_blended = np.clip(test_blended, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = test_blended
submission.to_csv('output/submission_phase2.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase2.csv 생성 완료")
print(submission.describe())

# ============================================================
# 6. 피처 중요도 (LightGBM 기준)
# ============================================================
print("\n=== 피처 중요도 (LightGBM) ===", flush=True)
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.mean([m.feature_importances_ for m in lgb_models], axis=0)
}).sort_values('importance', ascending=False)

top30 = importance.head(30).sort_values('importance')
ts_suffixes = ['_lag', '_roll', '_diff1', '_cumsum']
colors = ['coral' if any(s in f for s in ts_suffixes) else 'steelblue' for f in top30['feature']]

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color=colors)
ax.set_title('피처 중요도 Top 30 (Phase 2)')
ax.set_xlabel('중요도')
ax.legend(handles=[Patch(color='coral', label='시계열 피처'), Patch(color='steelblue', label='기존 피처')])
plt.tight_layout()
plt.savefig('output/feature_importance_phase2.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase2.png 저장 완료")

print("\n=== Phase 2 완료 ===", flush=True)
