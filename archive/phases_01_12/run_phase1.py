import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 데이터 로드 ===")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# === 1. 시계열 피처 생성 ===
print("\n=== 시계열 피처 생성 ===")
train['_is_train'] = 1
test['_is_train'] = 0
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined['implicit_timeslot'] = combined.groupby('scenario_id').cumcount()
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)

ts_cols = [
    'battery_mean', 'low_battery_ratio', 'order_inflow_15m', 'congestion_score',
    'robot_idle', 'robot_charging', 'max_zone_density', 'pack_utilization'
]

# Lag (1,2,3)
for col in ts_cols:
    for lag in [1, 2, 3]:
        combined[f'{col}_lag{lag}'] = combined.groupby('scenario_id')[col].shift(lag)

# Rolling (lag 기반)
for col in ts_cols:
    lag1 = combined[f'{col}_lag1']
    lag2 = combined[f'{col}_lag2']
    lag3 = combined[f'{col}_lag3']
    combined[f'{col}_roll3_mean'] = pd.concat([lag1, lag2, lag3], axis=1).mean(axis=1)
    combined[f'{col}_roll3_std'] = pd.concat([lag1, lag2, lag3], axis=1).std(axis=1)
    lag4 = combined.groupby('scenario_id')[col].shift(4)
    lag5 = combined.groupby('scenario_id')[col].shift(5)
    combined[f'{col}_roll5_mean'] = pd.concat([lag1, lag2, lag3, lag4, lag5], axis=1).mean(axis=1)
    combined[f'{col}_roll5_std'] = pd.concat([lag1, lag2, lag3, lag4, lag5], axis=1).std(axis=1)

# Diff
diff_cols = ['battery_mean', 'order_inflow_15m', 'congestion_score', 'robot_idle']
for col in diff_cols:
    combined[f'{col}_diff1'] = combined[col] - combined[f'{col}_lag1']

# Cumsum
cumsum_cols = ['order_inflow_15m', 'fault_count_15m', 'near_collision_15m', 'blocked_path_15m']
for col in cumsum_cols:
    combined[f'{col}_cumsum'] = combined.groupby('scenario_id')[col].cumsum()

ts_feature_names = [c for c in combined.columns if any(s in c for s in ['_lag', '_roll', '_diff1', '_cumsum'])]
print(f"생성된 시계열 피처 수: {len(ts_feature_names)}")

# === 2. 기존 피처 결합 ===
print("\n=== 기존 피처 결합 ===")
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

# 결측 지시자
train_part = combined[combined['_is_train'] == 1]
missing_counts = train_part.isnull().sum().sort_values(ascending=False)
top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith('_')]
for col in top10_missing:
    if col in combined.columns:
        combined[f'{col}_missing'] = combined[col].isnull().astype(int)

train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()

drop_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train']
feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
print(f"총 피처 수: {len(feature_cols)}")

# === 3. LightGBM 학습 ===
print("\n=== LightGBM 학습 (log1p target) ===")
X = train_fe[feature_cols]
y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y)
groups = train_fe['scenario_id']
X_test = test_fe[feature_cols]

params = {
    'objective': 'mae',
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

gkf = GroupKFold(n_splits=5)
fold_maes = []
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]
    y_val_orig = y.iloc[val_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )

    val_pred_log = model.predict(X_val)
    val_pred = np.clip(np.expm1(val_pred_log), 0, None)
    oof_preds[val_idx] = val_pred

    fold_mae = mean_absolute_error(y_val_orig, val_pred)
    fold_maes.append(fold_mae)

    test_pred_log = model.predict(X_test)
    test_preds += np.clip(np.expm1(test_pred_log), 0, None) / 5
    models.append(model)

    print(f"Fold {fold+1} MAE: {fold_mae:.4f} (best_iter: {model.best_iteration_})")

overall_mae = mean_absolute_error(y, oof_preds)
print(f"\n===== Phase 1 CV MAE: {overall_mae:.4f} =====")
print(f"Fold MAEs: {[f'{m:.4f}' for m in fold_maes]}")

baseline_mae = 9.182
improvement = baseline_mae - overall_mae
print(f"\n베이스라인 MAE: {baseline_mae:.3f} → 현재 MAE: {overall_mae:.4f} (개선: {improvement:.4f})")

# === 4. 피처 중요도 ===
print("\n=== 피처 중요도 분석 ===")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.mean([m.feature_importances_ for m in models], axis=0)
}).sort_values('importance', ascending=False)

top30 = importance.head(30)
ts_suffixes = ['_lag', '_roll', '_diff1', '_cumsum']
ts_in_top30 = top30[top30['feature'].apply(lambda x: any(s in x for s in ts_suffixes))]
print(f"Top 30 피처 중 시계열 피처: {len(ts_in_top30)}개")
print(ts_in_top30[['feature', 'importance']].to_string(index=False))

top30_plot = top30.sort_values('importance')
colors = ['coral' if any(s in f for s in ts_suffixes) else 'steelblue' for f in top30_plot['feature']]
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30_plot['feature'], top30_plot['importance'], color=colors)
ax.set_title('피처 중요도 Top 30 (Phase 1: 시계열 피처 추가)')
ax.set_xlabel('중요도')
ax.legend(handles=[Patch(color='coral', label='시계열 피처'), Patch(color='steelblue', label='기존 피처')])
plt.tight_layout()
plt.savefig('output/feature_importance_phase1.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase1.png 저장 완료")

# === 5. 제출 파일 ===
print("\n=== 제출 파일 생성 ===")
submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = np.clip(test_preds, 0, None)
submission.to_csv('output/submission_phase1.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns)
assert len(submission) == len(sample_sub)
assert submission['ID'].equals(sample_sub['ID'])
assert (submission['avg_delay_minutes_next_30m'] >= 0).all()

print("submission_phase1.csv 생성 완료")
print(submission.describe())
print("\n=== 완료 ===")
