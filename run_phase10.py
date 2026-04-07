import pandas as pd
import numpy as np
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 준비 (Phase 8 로직 그대로 — 319개 피처)
# ============================================================
print("=== 데이터 로드 ===", flush=True)
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

train['_is_train'] = 1
test['_is_train'] = 0
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined['_original_idx'] = range(len(combined))
del train, test
gc.collect()

combined['implicit_timeslot'] = combined.groupby('scenario_id').cumcount()
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)

# --- 시계열 피처 (64개) ---
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
print("  시계열 피처 64개 완료", flush=True)

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

# --- Phase 3B 인터랙션 (8개) ---
combined['orders_per_packstation'] = combined['order_inflow_15m'] / combined['pack_station_count'].replace(0, np.nan)
combined['pack_dock_pressure'] = combined['pack_utilization'] * combined['loading_dock_util']
combined['dock_wait_pressure'] = combined['outbound_truck_wait_min'] * combined['loading_dock_util']
combined['shift_load_pressure'] = combined['prev_shift_volume'] * combined['order_inflow_15m']
combined['battery_congestion'] = combined['low_battery_ratio'] * combined['congestion_score']
combined['storage_density_congestion'] = combined['storage_density_pct'] * combined['congestion_score']
combined['battery_trip_pressure'] = combined['low_battery_ratio'] * combined['avg_trip_distance']
combined['demand_density'] = combined['order_inflow_15m'] * combined['max_zone_density']

# --- 결측 지시자 ---
train_part = combined[combined['_is_train'] == 1]
missing_counts = train_part.isnull().sum().sort_values(ascending=False)
top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith('_')]
for col in top10_missing:
    if col in combined.columns:
        combined[f'{col}_missing'] = combined[col].isnull().astype(int)
del train_part
gc.collect()

# --- Onset 피처 (8개) ---
print("=== Onset 피처 ===", flush=True)

def compute_onset_idx(group, flag_col, timeslot_col='implicit_timeslot'):
    flags = group[flag_col].values
    slots = group[timeslot_col].values
    result = np.full(len(group), -1, dtype=np.float32)
    first_idx = -1
    for i in range(len(flags)):
        if i == 0:
            continue
        if flags[i-1] > 0 and first_idx == -1:
            first_idx = slots[i-1]
        result[i] = first_idx
    return pd.Series(result, index=group.index)

combined['_charging_flag'] = (combined['robot_charging'] > 0).astype(np.float32)
combined['_charging_flag_prev'] = combined.groupby('scenario_id')['_charging_flag'].shift(1).fillna(0)
combined['charging_ever_started'] = combined.groupby('scenario_id')['_charging_flag_prev'].cummax()
combined['charging_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_charging_flag'))
combined['charging_steps_since_start'] = np.where(
    combined['charging_start_idx'] >= 0,
    combined['implicit_timeslot'] - combined['charging_start_idx'], -1).astype(np.float32)
combined['charging_started_early'] = (
    (combined['charging_start_idx'] >= 0) & (combined['charging_start_idx'] < 5)).astype(np.float32)

combined['_queue_flag'] = (combined['charge_queue_length'] > 0).astype(np.float32)
combined['_queue_flag_prev'] = combined.groupby('scenario_id')['_queue_flag'].shift(1).fillna(0)
combined['queue_ever_started'] = combined.groupby('scenario_id')['_queue_flag_prev'].cummax()
combined['queue_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_queue_flag'))

combined['_congestion_flag'] = (combined['congestion_score'] > 0).astype(np.float32)
combined['_congestion_flag_prev'] = combined.groupby('scenario_id')['_congestion_flag'].shift(1).fillna(0)
combined['congestion_ever_started'] = combined.groupby('scenario_id')['_congestion_flag_prev'].cummax()
combined['congestion_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_congestion_flag'))

combined.drop(columns=['_charging_flag', '_charging_flag_prev', '_queue_flag', '_queue_flag_prev',
                        '_congestion_flag', '_congestion_flag_prev'], inplace=True)
print("  Onset 8개 완료", flush=True)

# --- Expanding Mean (30개) ---
print("=== Expanding Mean ===", flush=True)
expanding_cols_mean = [
    'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
    'heavy_item_ratio', 'robot_active', 'battery_mean', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'pack_utilization', 'loading_dock_util',
    'charge_queue_length', 'fault_count_15m', 'avg_trip_distance'
]
for i, col in enumerate(expanding_cols_mean):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    expmean = shifted.groupby(combined['scenario_id']).expanding().mean().droplevel(0).sort_index()
    combined[f'{col}_expmean_prev'] = expmean.astype(np.float32)
    combined[f'{col}_delta_expmean'] = (combined[col] - combined[f'{col}_expmean_prev']).astype(np.float32)
    if (i + 1) % 5 == 0:
        print(f"  Expanding Mean: {i+1}/{len(expanding_cols_mean)}", flush=True)

# --- 비선형 (7개) ---
combined['battery_mean_below_44'] = np.maximum(44.0 - combined['battery_mean'], 0).astype(np.float32)
combined['low_battery_ratio_above_02'] = np.maximum(combined['low_battery_ratio'] - 0.2, 0).astype(np.float32)
combined['pack_utilization_sq'] = (combined['pack_utilization'] ** 2).astype(np.float32)
combined['loading_dock_util_sq'] = (combined['loading_dock_util'] ** 2).astype(np.float32)
combined['congestion_score_sq'] = (combined['congestion_score'] ** 2).astype(np.float32)
combined['charge_pressure_nl'] = ((combined['robot_charging'] + combined['charge_queue_length']) / (combined['charger_count'] + 1)).astype(np.float32)
combined['charge_pressure_nl_sq'] = (combined['charge_pressure_nl'] ** 2).astype(np.float32)

# --- 위상 (6개) ---
combined['is_early_phase'] = (combined['implicit_timeslot'] <= 5).astype(np.float32)
combined['is_mid_phase'] = ((combined['implicit_timeslot'] >= 6) & (combined['implicit_timeslot'] <= 15)).astype(np.float32)
combined['is_late_phase'] = (combined['implicit_timeslot'] >= 16).astype(np.float32)
combined['time_frac'] = (combined['implicit_timeslot'] / 24.0).astype(np.float32)
combined['time_remaining'] = (24 - combined['implicit_timeslot']).astype(np.float32)
combined['time_frac_sq'] = (combined['time_frac'] ** 2).astype(np.float32)

# --- 경쟁자 피처 (54개) ---
print("=== 경쟁자 피처 ===", flush=True)
combined['robot_total_state'] = combined['robot_active'] + combined['robot_idle'] + combined['robot_charging']
combined['robot_total_gap'] = combined['robot_total_state'] - combined['robot_total']
combined['robot_active_share'] = combined['robot_active'] / (combined['robot_total_state'] + 1)
combined['robot_idle_share'] = combined['robot_idle'] / (combined['robot_total_state'] + 1)
combined['robot_charging_share'] = combined['robot_charging'] / (combined['robot_total_state'] + 1)
combined['charging_to_active_ratio'] = combined['robot_charging'] / (combined['robot_active'] + 1)

combined['inflow_per_robot'] = combined['order_inflow_15m'] / (combined['robot_total'] + 1)
combined['inflow_per_pack_station'] = combined['order_inflow_15m'] / (combined['pack_station_count'] + 1)
combined['unique_sku_per_robot'] = combined['unique_sku_15m'] / (combined['robot_total'] + 1)
combined['unique_sku_per_pack_station'] = combined['unique_sku_15m'] / (combined['pack_station_count'] + 1)
combined['charge_queue_per_charger'] = combined['charge_queue_length'] / (combined['charger_count'] + 1)
combined['charging_per_charger'] = combined['robot_charging'] / (combined['charger_count'] + 1)
combined['congestion_per_width'] = combined['congestion_score'] / (combined['aisle_width_avg'] + 0.01)
combined['zone_density_per_width'] = combined['max_zone_density'] / (combined['aisle_width_avg'] + 0.01)
combined['order_per_sqm'] = combined['order_inflow_15m'] / (combined['floor_area_sqm'] + 1)
combined['dock_pressure'] = combined['order_inflow_15m'] / (combined['staff_on_floor'] + 1)
combined['fault_per_active'] = combined['fault_count_15m'] / (combined['robot_active'] + 1)
combined['collision_per_active'] = combined['near_collision_15m'] / (combined['robot_active'] + 1)
combined['blocked_per_active'] = combined['blocked_path_15m'] / (combined['robot_active'] + 1)
combined['congestion_per_active'] = combined['congestion_score'] / (combined['robot_active'] + 1)
combined['label_queue_per_pack'] = combined['label_print_queue'] / (combined['pack_station_count'] + 1)

combined['demand_mass'] = combined['order_inflow_15m'] * combined['avg_package_weight_kg']
combined['demand_mass_per_robot'] = combined['demand_mass'] / (combined['robot_total'] + 1)
combined['trip_load'] = combined['order_inflow_15m'] * combined['avg_trip_distance']
combined['trip_load_per_robot'] = combined['trip_load'] / (combined['robot_total'] + 1)
combined['complexity_load'] = combined['order_inflow_15m'] * combined['unique_sku_15m']
combined['complexity_load_per_pack'] = combined['complexity_load'] / (combined['pack_station_count'] + 1)
combined['congestion_x_lowbat'] = combined['congestion_score'] * combined['low_battery_ratio']
combined['battery_pressure'] = combined['low_battery_ratio'] * combined['robot_active']
combined['queue_wait_pressure'] = combined['charge_queue_length'] * combined['avg_charge_wait']
combined['dock_pack_pressure'] = combined['loading_dock_util'] * combined['pack_utilization']
combined['staging_pack_pressure'] = combined['staging_area_util'] * combined['pack_utilization']
combined['charge_pressure'] = (combined['robot_charging'] + combined['charge_queue_length']) / (combined['charger_count'] + 1)

combined['warehouse_volume'] = combined['floor_area_sqm'] * combined['ceiling_height_m']
combined['intersection_density'] = combined['intersection_count'] / (combined['floor_area_sqm'] + 1)
combined['pack_station_density'] = combined['pack_station_count'] / (combined['floor_area_sqm'] + 1)
combined['charger_density'] = combined['charger_count'] / (combined['floor_area_sqm'] + 1)
combined['robot_density_layout'] = combined['robot_total'] / (combined['floor_area_sqm'] + 1)
combined['movement_friction'] = combined['intersection_count'] / (combined['aisle_width_avg'] + 0.01)
combined['layout_compact_x_dispersion'] = combined['layout_compactness'] * combined['zone_dispersion']
combined['one_way_friction'] = combined['one_way_ratio'] * combined['intersection_count'] / (combined['aisle_width_avg'] + 0.01)

numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if not c.startswith('_')]
combined['n_missing_all'] = combined[numeric_cols].isnull().sum(axis=1).astype(np.float32)
dynamic_cols = ['order_inflow_15m', 'battery_mean', 'battery_std', 'low_battery_ratio',
                'robot_active', 'robot_idle', 'robot_charging', 'congestion_score',
                'max_zone_density', 'pack_utilization', 'loading_dock_util',
                'charge_queue_length', 'fault_count_15m', 'avg_trip_distance', 'unique_sku_15m']
dynamic_cols_exist = [c for c in dynamic_cols if c in combined.columns]
combined['n_missing_dynamic'] = combined[dynamic_cols_exist].isnull().sum(axis=1).astype(np.float32)
combined['missing_ratio'] = (combined['n_missing_all'] / len(numeric_cols)).astype(np.float32)

print("  rolling max + 편차...", flush=True)
rollmax_cols = ['order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization', 'loading_dock_util']
for col in rollmax_cols:
    shifted = combined.groupby('scenario_id')[col].shift(1)
    rollmax3 = shifted.groupby(combined['scenario_id']).rolling(3, min_periods=1).max().droplevel(0).sort_index()
    combined[f'{col}_rollmax3_prev'] = rollmax3.astype(np.float32)
    combined[f'{col}_dev_rollmax3'] = (combined[col] - combined[f'{col}_rollmax3_prev']).astype(np.float32)

# --- Expanding 확장 B (20개) ---
print("=== Expanding 확장 (std, max) ===", flush=True)
expanding_ext_cols = [
    'order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization',
    'loading_dock_util', 'robot_active', 'low_battery_ratio', 'avg_trip_distance',
    'unique_sku_15m', 'max_zone_density'
]
for i, col in enumerate(expanding_ext_cols):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    grp = shifted.groupby(combined['scenario_id'])
    combined[f'{col}_expstd_prev'] = grp.expanding().std().droplevel(0).sort_index().astype(np.float32)
    combined[f'{col}_expmax_prev'] = grp.expanding().max().droplevel(0).sort_index().astype(np.float32)
    if (i + 1) % 5 == 0:
        print(f"  Expanding 확장: {i+1}/{len(expanding_ext_cols)}", flush=True)

print("  피처 생성 완료", flush=True)

# --- float32 + 분리 ---
print("\n=== float32 변환 + 분리 ===", flush=True)
float64_cols = combined.select_dtypes(include='float64').columns
combined[float64_cols] = combined[float64_cols].astype(np.float32)

combined = combined.sort_values('_original_idx').reset_index(drop=True)
train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()
del combined
gc.collect()

drop_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']
feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
print(f"총 피처 수: {len(feature_cols)}", flush=True)

X = train_fe[feature_cols]
y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y)
y_sqrt = np.sqrt(y)
groups = train_fe['scenario_id']
time_idx = train_fe['implicit_timeslot'].values.astype(np.float32)
X_test = test_fe[feature_cols]

assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID 순서 불일치!"
print("ID 순서 검증 통과!", flush=True)

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y_log, groups))

# --- 샘플 가중치 ---
def build_sample_weight(y_arr, time_arr):
    w = np.ones(len(y_arr), dtype=np.float32)
    q90 = np.nanquantile(y_arr, 0.90)
    q95 = np.nanquantile(y_arr, 0.95)
    q99 = np.nanquantile(y_arr, 0.99)
    w += 0.15 * (y_arr >= q90).astype(np.float32)
    w += 0.30 * (y_arr >= q95).astype(np.float32)
    w += 0.60 * (y_arr >= q99).astype(np.float32)
    if time_arr is not None:
        w += 0.08 * (time_arr / 24.0).astype(np.float32)
    return w

sample_w = build_sample_weight(y.values, time_idx)
print(f"  가중치: min={sample_w.min():.2f}, max={sample_w.max():.2f}, mean={sample_w.mean():.2f}", flush=True)

# ============================================================
# 2. NN 전용 전처리
# ============================================================
print("\n=== NN 전용 전처리 ===", flush=True)
scaler = StandardScaler()
X_train_nn = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_test_nn = np.nan_to_num(X_test.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_train_nn = scaler.fit_transform(X_train_nn).astype(np.float32)
X_test_nn = scaler.transform(X_test_nn).astype(np.float32)
y_log_nn = y_log.values.astype(np.float32)
print(f"  NN 데이터: train {X_train_nn.shape}, test {X_test_nn.shape}", flush=True)

# ============================================================
# 3. 트리 6모델 학습 (Phase 8 동일)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== 트리 6모델 학습 ===", flush=True)
print("=" * 60, flush=True)

oof_preds = {}
test_preds = {}
cv_maes = {}

# 모델 1: LightGBM raw+MAE (Optuna params)
print("\n  [모델 1] LGB raw+MAE...", flush=True)
m1_oof = np.zeros(len(X), dtype=np.float32)
m1_test = np.zeros(len(X_test), dtype=np.float32)
m1_models = []
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=2000, learning_rate=0.0129,
        num_leaves=185, max_depth=9, min_child_samples=80,
        reg_alpha=0.0574, reg_lambda=0.0042,
        feature_fraction=0.6005, bagging_fraction=0.7663, bagging_freq=1,
        random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sample_w[tr_idx],
          eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(m.predict(X.iloc[va_idx]), 0, None).astype(np.float32)
    m1_oof[va_idx] = pred
    m1_test += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
    m1_models.append(m)
    print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
oof_preds['lgb_raw_mae'] = m1_oof
test_preds['lgb_raw_mae'] = m1_test
cv_maes['lgb_raw_mae'] = mean_absolute_error(y, m1_oof)
print(f"  모델 1 CV MAE: {cv_maes['lgb_raw_mae']:.4f}", flush=True)

# 모델 2: LightGBM log1p+Huber
print("\n  [모델 2] LGB log1p+Huber...", flush=True)
m2_oof = np.zeros(len(X), dtype=np.float32)
m2_test = np.zeros(len(X_test), dtype=np.float32)
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(
        objective='huber', huber_delta=0.9, n_estimators=2000, learning_rate=0.03,
        num_leaves=128, min_child_samples=60, subsample=0.9, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr_idx], y_log.iloc[tr_idx], sample_weight=sample_w[tr_idx],
          eval_set=[(X.iloc[va_idx], y_log.iloc[va_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m.predict(X.iloc[va_idx])), 0, None).astype(np.float32)
    m2_oof[va_idx] = pred
    m2_test += np.clip(np.expm1(m.predict(X_test)), 0, None).astype(np.float32) / 5
    print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
oof_preds['lgb_log1p_huber'] = m2_oof
test_preds['lgb_log1p_huber'] = m2_test
cv_maes['lgb_log1p_huber'] = mean_absolute_error(y, m2_oof)
print(f"  모델 2 CV MAE: {cv_maes['lgb_log1p_huber']:.4f}", flush=True)

# 모델 3: LightGBM sqrt+MAE
print("\n  [모델 3] LGB sqrt+MAE...", flush=True)
m3_oof = np.zeros(len(X), dtype=np.float32)
m3_test = np.zeros(len(X_test), dtype=np.float32)
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=2000, learning_rate=0.03,
        num_leaves=96, min_child_samples=80, subsample=0.9, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr_idx], y_sqrt.iloc[tr_idx], sample_weight=sample_w[tr_idx],
          eval_set=[(X.iloc[va_idx], y_sqrt.iloc[va_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(m.predict(X.iloc[va_idx]) ** 2, 0, None).astype(np.float32)
    m3_oof[va_idx] = pred
    m3_test += np.clip(m.predict(X_test) ** 2, 0, None).astype(np.float32) / 5
    print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
oof_preds['lgb_sqrt_mae'] = m3_oof
test_preds['lgb_sqrt_mae'] = m3_test
cv_maes['lgb_sqrt_mae'] = mean_absolute_error(y, m3_oof)
print(f"  모델 3 CV MAE: {cv_maes['lgb_sqrt_mae']:.4f}", flush=True)

# 모델 4: XGBoost raw+MAE
print("\n  [모델 4] XGB raw+MAE...", flush=True)
m4_oof = np.zeros(len(X), dtype=np.float32)
m4_test = np.zeros(len(X_test), dtype=np.float32)
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        min_child_weight=6, subsample=0.9, colsample_bytree=0.85,
        reg_lambda=1.5, reg_alpha=0.05,
        objective='reg:absoluteerror', eval_metric='mae',
        tree_method='hist', random_state=42, verbosity=0, early_stopping_rounds=100)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sample_w[tr_idx],
          eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
    pred = np.clip(m.predict(X.iloc[va_idx]), 0, None).astype(np.float32)
    m4_oof[va_idx] = pred
    m4_test += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
    print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
oof_preds['xgb_raw_mae'] = m4_oof
test_preds['xgb_raw_mae'] = m4_test
cv_maes['xgb_raw_mae'] = mean_absolute_error(y, m4_oof)
print(f"  모델 4 CV MAE: {cv_maes['xgb_raw_mae']:.4f}", flush=True)

# 모델 5: CatBoost log1p+MAE
print("\n  [모델 5] Cat log1p+MAE...", flush=True)
m5_oof = np.zeros(len(X), dtype=np.float32)
m5_test = np.zeros(len(X_test), dtype=np.float32)
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    train_pool = Pool(X.iloc[tr_idx], y_log.iloc[tr_idx], weight=sample_w[tr_idx])
    eval_pool = Pool(X.iloc[va_idx], y_log.iloc[va_idx])
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=8,
        l2_leaf_reg=5.0, subsample=0.9,
        loss_function='MAE', random_seed=42, verbose=0)
    m.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=100)
    pred = np.clip(np.expm1(m.predict(X.iloc[va_idx])), 0, None).astype(np.float32)
    m5_oof[va_idx] = pred
    m5_test += np.clip(np.expm1(m.predict(X_test)), 0, None).astype(np.float32) / 5
    print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
oof_preds['cat_log1p_mae'] = m5_oof
test_preds['cat_log1p_mae'] = m5_test
cv_maes['cat_log1p_mae'] = mean_absolute_error(y, m5_oof)
print(f"  모델 5 CV MAE: {cv_maes['cat_log1p_mae']:.4f}", flush=True)

# 모델 6: CatBoost raw+MAE
print("\n  [모델 6] Cat raw+MAE...", flush=True)
m6_oof = np.zeros(len(X), dtype=np.float32)
m6_test = np.zeros(len(X_test), dtype=np.float32)
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], weight=sample_w[tr_idx])
    eval_pool = Pool(X.iloc[va_idx], y.iloc[va_idx])
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, subsample=0.85,
        loss_function='MAE', random_seed=42, verbose=0)
    m.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=100)
    pred = np.clip(m.predict(X.iloc[va_idx]), 0, None).astype(np.float32)
    m6_oof[va_idx] = pred
    m6_test += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
    print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
oof_preds['cat_raw_mae'] = m6_oof
test_preds['cat_raw_mae'] = m6_test
cv_maes['cat_raw_mae'] = mean_absolute_error(y, m6_oof)
print(f"  모델 6 CV MAE: {cv_maes['cat_raw_mae']:.4f}", flush=True)

# ============================================================
# 4. NN 모델 1: Keras MLP
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== NN 모델: Keras MLP ===", flush=True)
print("=" * 60, flush=True)

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks as keras_callbacks

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

mlp_oof = np.zeros(len(X), dtype=np.float32)
mlp_test = np.zeros(len(X_test), dtype=np.float32)

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    print(f"  MLP Fold {fold_i+1}/5...", flush=True)
    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_mlp(X_train_nn.shape[1])
    es = keras_callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    rlr = keras_callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)

    model.fit(
        X_train_nn[tr_idx], y_log_nn[tr_idx],
        validation_data=(X_train_nn[va_idx], y_log_nn[va_idx]),
        epochs=100, batch_size=512, callbacks=[es, rlr], verbose=0
    )

    pred_log = model.predict(X_train_nn[va_idx], verbose=0).flatten()
    pred = np.clip(np.expm1(pred_log), 0, None).astype(np.float32)
    mlp_oof[va_idx] = pred
    mlp_test += np.clip(np.expm1(model.predict(X_test_nn, verbose=0).flatten()), 0, None).astype(np.float32) / 5

    mae = mean_absolute_error(y.values[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f}", flush=True)

    del model
    tf.keras.backend.clear_session()
    gc.collect()

mlp_cv = mean_absolute_error(y, mlp_oof)
oof_preds['keras_mlp'] = mlp_oof
test_preds['keras_mlp'] = mlp_test
cv_maes['keras_mlp'] = mlp_cv
print(f"  Keras MLP CV MAE: {mlp_cv:.4f}", flush=True)

# ============================================================
# 5. NN 모델 2: TabNet
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== NN 모델: TabNet ===", flush=True)
print("=" * 60, flush=True)

from pytorch_tabnet.tab_model import TabNetRegressor
import torch

tabnet_oof = np.zeros(len(X), dtype=np.float32)
tabnet_test = np.zeros(len(X_test), dtype=np.float32)

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    print(f"  TabNet Fold {fold_i+1}/5...", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)

    model = TabNetRegressor(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        n_independent=2, n_shared=2, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax', seed=42, verbose=10,
    )

    model.fit(
        X_train_nn[tr_idx], y_log_nn[tr_idx].reshape(-1, 1),
        eval_set=[(X_train_nn[va_idx], y_log_nn[va_idx].reshape(-1, 1))],
        eval_metric=['mae'],
        max_epochs=100, patience=15,
        batch_size=2048, virtual_batch_size=256,
    )

    pred_log = model.predict(X_train_nn[va_idx]).flatten()
    pred = np.clip(np.expm1(pred_log), 0, None).astype(np.float32)
    tabnet_oof[va_idx] = pred
    tabnet_test += np.clip(np.expm1(model.predict(X_test_nn).flatten()), 0, None).astype(np.float32) / 5

    mae = mean_absolute_error(y.values[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f}", flush=True)

    del model
    gc.collect()

tabnet_cv = mean_absolute_error(y, tabnet_oof)
oof_preds['tabnet'] = tabnet_oof
test_preds['tabnet'] = tabnet_test
cv_maes['tabnet'] = tabnet_cv
print(f"  TabNet CV MAE: {tabnet_cv:.4f}", flush=True)

# ============================================================
# 6. Level 1 가중 평균 앙상블
# ============================================================
print("\n=== Level 1 가중 평균 앙상블 ===", flush=True)

model_names = list(oof_preds.keys())
tree_names = [n for n in model_names if n not in ('keras_mlp', 'tabnet')]
oof_matrix = np.column_stack([oof_preds[n] for n in model_names])
test_matrix = np.column_stack([test_preds[n] for n in model_names])

# 트리만
tree_oof_matrix = np.column_stack([oof_preds[n] for n in tree_names])
tree_test_matrix = np.column_stack([test_preds[n] for n in tree_names])

def opt_weights(oof_mat, y_true):
    n = oof_mat.shape[1]
    def obj(w):
        w = w / w.sum()
        return mean_absolute_error(y_true, oof_mat @ w)
    res = minimize(obj, x0=np.ones(n) / n, method='Nelder-Mead', options={'maxiter': 10000})
    w = res.x / res.x.sum()
    return w, mean_absolute_error(y_true, oof_mat @ w)

# 트리만 앙상블
tree_w, tree_cv = opt_weights(tree_oof_matrix, y)
print("  트리만 가중치:", flush=True)
for n, w in zip(tree_names, tree_w):
    print(f"    {n:20s}: {w:.4f}", flush=True)
print(f"  트리만 Level 1 CV MAE: {tree_cv:.4f}", flush=True)

# 전체 8모델 앙상블
all_w, all_cv = opt_weights(oof_matrix, y)
print("\n  트리+NN 가중치:", flush=True)
for n, w in zip(model_names, all_w):
    print(f"    {n:20s}: {w:.4f}", flush=True)
print(f"  트리+NN Level 1 CV MAE: {all_cv:.4f}", flush=True)

# ============================================================
# 7. Level 2 LGB 스태킹 (8모델)
# ============================================================
print("\n=== Level 2 LGB 스태킹 (8모델) ===", flush=True)

meta_cols_extra = ['implicit_timeslot', 'order_inflow_15m', 'battery_mean',
                   'robot_active', 'pack_utilization', 'congestion_score']
meta_train_extra = train_fe[meta_cols_extra].values.astype(np.float32)
meta_test_extra = test_fe[meta_cols_extra].values.astype(np.float32)

meta_X_train = np.nan_to_num(np.column_stack([oof_matrix, meta_train_extra]), nan=0.0)
meta_X_test = np.nan_to_num(np.column_stack([test_matrix, meta_test_extra]), nan=0.0)
meta_feature_names = model_names + meta_cols_extra
print(f"  메타 피처 수: {len(meta_feature_names)}", flush=True)

meta_X_train_df = pd.DataFrame(meta_X_train, columns=meta_feature_names)
meta_X_test_df = pd.DataFrame(meta_X_test, columns=meta_feature_names)

meta_lgb_oof = np.zeros(len(X), dtype=np.float32)
meta_lgb_test = np.zeros(len(meta_X_test), dtype=np.float32)

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=16, max_depth=4,
        objective='mae', random_state=42, n_jobs=-1, verbose=-1)
    m.fit(meta_X_train_df.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(meta_X_train_df.iloc[va_idx], y.iloc[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(m.predict(meta_X_train_df.iloc[va_idx]), 0, None).astype(np.float32)
    meta_lgb_oof[va_idx] = pred
    meta_lgb_test += np.clip(m.predict(meta_X_test_df), 0, None).astype(np.float32) / 5
    print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)

meta_lgb_cv = mean_absolute_error(y, meta_lgb_oof)
print(f"  Level 2 LGB 스태킹 CV MAE: {meta_lgb_cv:.4f}", flush=True)

# --- 최종 선택 ---
candidates = {
    'tree_only_l1': (tree_cv, tree_oof_matrix @ tree_w, tree_test_matrix @ tree_w),
    'all_l1': (all_cv, oof_matrix @ all_w, test_matrix @ all_w),
    'lgb_stack': (meta_lgb_cv, meta_lgb_oof, meta_lgb_test),
}

print("\n  비교:", flush=True)
for name, (cv, _, _) in candidates.items():
    print(f"    {name:20s}: {cv:.4f}", flush=True)

best_name = min(candidates, key=lambda k: candidates[k][0])
final_cv = candidates[best_name][0]
final_test = candidates[best_name][2]
print(f"\n  최종 선택: {best_name} (CV MAE: {final_cv:.4f})", flush=True)

# ============================================================
# 8. 결과 비교
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 10 결과 ===", flush=True)
print("=" * 60, flush=True)
print("트리 모델:", flush=True)
for n in tree_names:
    print(f"  {n:20s}: {cv_maes[n]:.4f}", flush=True)
print("NN 모델:", flush=True)
print(f"  {'keras_mlp':20s}: {cv_maes['keras_mlp']:.4f}", flush=True)
print(f"  {'tabnet':20s}: {cv_maes['tabnet']:.4f}", flush=True)
print(f"Level 1 가중 평균 (트리만):  {tree_cv:.4f}", flush=True)
print(f"Level 1 가중 평균 (트리+NN): {all_cv:.4f}", flush=True)
print(f"Level 2 LGB 스태킹 (8모델): {meta_lgb_cv:.4f}", flush=True)
print(f"최종 선택: {best_name} ({final_cv:.4f})", flush=True)
print(f"Phase 8 대비 개선: {final_cv - 8.653:+.4f}", flush=True)

# ============================================================
# 9. 제출 파일
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
final_test = np.clip(final_test, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase10.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase10.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 10. 피처 중요도 (LGB raw+MAE)
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': m1_models[0].feature_importances_
}).sort_values('importance', ascending=False)

top30 = importance.head(30).sort_values('importance')

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color='steelblue')
ax.set_title('Feature Importance Top 30 (Phase 10 - LGB raw+MAE)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('output/feature_importance_phase10.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase10.png 저장 완료", flush=True)

print("\n=== Phase 10 완료 ===", flush=True)
