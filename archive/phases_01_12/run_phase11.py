import pandas as pd
import numpy as np
import gc
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.optimize import minimize
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 준비 (Phase 10 동일 — 319개 피처)
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

# --- Transformer용 원본 시계열 데이터 저장 (분리 전) ---
print("\n=== Transformer용 시계열 데이터 준비 ===", flush=True)
raw_ts_cols = [
    'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
    'heavy_item_ratio', 'cold_chain_ratio', 'sku_concentration',
    'robot_active', 'robot_idle', 'robot_charging', 'robot_utilization',
    'avg_trip_distance', 'task_reassign_15m',
    'battery_mean', 'battery_std', 'low_battery_ratio', 'charge_queue_length', 'avg_charge_wait',
    'congestion_score', 'max_zone_density', 'blocked_path_15m', 'near_collision_15m',
    'fault_count_15m', 'avg_recovery_time', 'replenishment_overlap',
    'pack_utilization', 'manual_override_ratio',
    'warehouse_temp_avg', 'humidity_pct', 'day_of_week',
    'external_temp_c', 'wind_speed_kmh', 'precipitation_mm',
    'lighting_level_lux', 'ambient_noise_db', 'floor_vibration_idx',
    'return_order_ratio', 'air_quality_idx', 'co2_level_ppm', 'hvac_power_kw',
    'wms_response_time_ms', 'scanner_error_rate', 'wifi_signal_db', 'network_latency_ms',
    'worker_avg_tenure_months', 'safety_score_monthly',
    'label_print_queue', 'barcode_read_success_rate', 'ups_battery_pct', 'lighting_zone_variance',
    'shift_hour', 'staff_on_floor', 'forklift_active_count',
    'loading_dock_util', 'conveyor_speed_mps', 'prev_shift_volume',
    'avg_package_weight_kg', 'inventory_turnover_rate', 'daily_forecast_accuracy',
    'order_wave_count', 'pick_list_length_avg', 'express_lane_util', 'bulk_order_ratio',
    'staging_area_util', 'cold_storage_temp_c', 'pallet_wrap_time_min',
    'fleet_age_months_avg', 'maintenance_schedule_score', 'robot_firmware_update_days',
    'avg_idle_duration_min', 'charge_efficiency_pct', 'battery_cycle_count_avg',
    'agv_task_success_rate', 'robot_calibration_score',
    'aisle_traffic_score', 'zone_temp_variance', 'path_optimization_score',
    'intersection_wait_time_avg', 'storage_density_pct', 'vertical_utilization',
    'racking_height_avg_m', 'cross_dock_ratio', 'packaging_material_cost',
    'quality_check_rate', 'outbound_truck_wait_min', 'dock_to_stock_hours',
    'kpi_otd_pct', 'backorder_ratio', 'shift_handover_delay_min', 'sort_accuracy_pct'
]
# Filter to columns that exist
raw_ts_cols = [c for c in raw_ts_cols if c in combined.columns]
n_raw_features = len(raw_ts_cols)
print(f"  Transformer 원본 피처 수: {n_raw_features}", flush=True)

# scenario_id와 implicit_timeslot, raw features 추출
ts_data = combined[['scenario_id', 'implicit_timeslot', '_is_train', '_original_idx'] + raw_ts_cols].copy()

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

# --- 기본 샘플 가중치 ---
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
print(f"  기본 가중치: min={sample_w.min():.2f}, max={sample_w.max():.2f}, mean={sample_w.mean():.2f}", flush=True)

# ============================================================
# 2. Adversarial Validation
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Adversarial Validation ===", flush=True)
print("=" * 60, flush=True)

X_train_vals = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_test_vals = np.nan_to_num(X_test.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

X_combined_adv = np.vstack([X_train_vals, X_test_vals])
y_adv = np.concatenate([np.zeros(len(X_train_vals)), np.ones(len(X_test_vals))])

adv_oof = np.zeros(len(X_combined_adv), dtype=np.float64)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_i, (tr, va) in enumerate(skf.split(X_combined_adv, y_adv)):
    m = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        max_depth=6, verbosity=-1, random_state=42, n_jobs=-1
    )
    m.fit(X_combined_adv[tr], y_adv[tr])
    adv_oof[va] = m.predict_proba(X_combined_adv[va])[:, 1]
    print(f"  Adv Fold {fold_i+1}/5 완료", flush=True)

auc = roc_auc_score(y_adv, adv_oof)
print(f"  Adversarial AUC: {auc:.4f}", flush=True)
print(f"  (0.5 = train/test 동일 분포, 1.0 = 완전히 다른 분포)", flush=True)

# train 부분의 "test 닮은 점수" 추출
train_test_score = adv_oof[:len(X_train_vals)]

# sample weight 생성: test와 닮은 샘플에 가중치 부여
adv_weight = 1.0 + 2.0 * train_test_score  # 1.0 ~ 3.0
combined_weight = (sample_w * adv_weight).astype(np.float32)
print(f"  Adv weight: min={adv_weight.min():.2f}, max={adv_weight.max():.2f}, mean={adv_weight.mean():.2f}", flush=True)
print(f"  Combined weight: min={combined_weight.min():.2f}, max={combined_weight.max():.2f}, mean={combined_weight.mean():.2f}", flush=True)

del X_combined_adv, adv_oof
gc.collect()

# ============================================================
# 3. NN 전용 전처리
# ============================================================
print("\n=== NN 전용 전처리 ===", flush=True)
scaler = StandardScaler()
X_train_nn = X_train_vals.copy()
X_test_nn = X_test_vals.copy()
X_train_nn = scaler.fit_transform(X_train_nn).astype(np.float32)
X_test_nn = scaler.transform(X_test_nn).astype(np.float32)
y_log_nn = y_log.values.astype(np.float32)
print(f"  NN 데이터: train {X_train_nn.shape}, test {X_test_nn.shape}", flush=True)

del X_train_vals, X_test_vals
gc.collect()

# ============================================================
# 4. 트리 6모델 학습 (combined_weight 사용)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== 트리 6모델 학습 (Adversarial Weight 적용) ===", flush=True)
print("=" * 60, flush=True)

oof_preds = {}
test_preds = {}
cv_maes = {}

# 모델 1: LightGBM raw+MAE
print("\n  [모델 1] LGB raw+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase11_lgb_raw.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    m1_oof, m1_test, cv_maes['lgb_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    m1_models = []
    print(f"  ⏭️ lgb_raw 캐시 사용 (CV MAE: {cv_maes['lgb_raw_mae']:.4f})", flush=True)
else:
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
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=combined_weight[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred = np.clip(m.predict(X.iloc[va_idx]), 0, None).astype(np.float32)
        m1_oof[va_idx] = pred
        m1_test += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
        m1_models.append(m)
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
    cv_maes['lgb_raw_mae'] = mean_absolute_error(y, m1_oof)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': m1_oof, 'test': m1_test, 'cv_mae': cv_maes['lgb_raw_mae']}, f)
    print(f"  ✅ lgb_raw 저장 완료 (CV MAE: {cv_maes['lgb_raw_mae']:.4f})", flush=True)
oof_preds['lgb_raw_mae'] = m1_oof
test_preds['lgb_raw_mae'] = m1_test
print(f"  모델 1 CV MAE: {cv_maes['lgb_raw_mae']:.4f}", flush=True)

# 모델 2: LightGBM log1p+Huber
print("\n  [모델 2] LGB log1p+Huber...", flush=True)
ckpt_path = 'output/ckpt_phase11_lgb_huber.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    m2_oof, m2_test, cv_maes['lgb_log1p_huber'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  ⏭️ lgb_huber 캐시 사용 (CV MAE: {cv_maes['lgb_log1p_huber']:.4f})", flush=True)
else:
    m2_oof = np.zeros(len(X), dtype=np.float32)
    m2_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            objective='huber', huber_delta=0.9, n_estimators=2000, learning_rate=0.03,
            num_leaves=128, min_child_samples=60, subsample=0.9, colsample_bytree=0.85,
            reg_alpha=0.05, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tr_idx], y_log.iloc[tr_idx], sample_weight=combined_weight[tr_idx],
              eval_set=[(X.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred = np.clip(np.expm1(m.predict(X.iloc[va_idx])), 0, None).astype(np.float32)
        m2_oof[va_idx] = pred
        m2_test += np.clip(np.expm1(m.predict(X_test)), 0, None).astype(np.float32) / 5
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
    cv_maes['lgb_log1p_huber'] = mean_absolute_error(y, m2_oof)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': m2_oof, 'test': m2_test, 'cv_mae': cv_maes['lgb_log1p_huber']}, f)
    print(f"  ✅ lgb_huber 저장 완료 (CV MAE: {cv_maes['lgb_log1p_huber']:.4f})", flush=True)
oof_preds['lgb_log1p_huber'] = m2_oof
test_preds['lgb_log1p_huber'] = m2_test
print(f"  모델 2 CV MAE: {cv_maes['lgb_log1p_huber']:.4f}", flush=True)

# 모델 3: LightGBM sqrt+MAE
print("\n  [모델 3] LGB sqrt+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase11_lgb_sqrt.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    m3_oof, m3_test, cv_maes['lgb_sqrt_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  ⏭️ lgb_sqrt 캐시 사용 (CV MAE: {cv_maes['lgb_sqrt_mae']:.4f})", flush=True)
else:
    m3_oof = np.zeros(len(X), dtype=np.float32)
    m3_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            objective='mae', n_estimators=2000, learning_rate=0.03,
            num_leaves=96, min_child_samples=80, subsample=0.9, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tr_idx], y_sqrt.iloc[tr_idx], sample_weight=combined_weight[tr_idx],
              eval_set=[(X.iloc[va_idx], y_sqrt.iloc[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred = np.clip(m.predict(X.iloc[va_idx]) ** 2, 0, None).astype(np.float32)
        m3_oof[va_idx] = pred
        m3_test += np.clip(m.predict(X_test) ** 2, 0, None).astype(np.float32) / 5
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
    cv_maes['lgb_sqrt_mae'] = mean_absolute_error(y, m3_oof)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': m3_oof, 'test': m3_test, 'cv_mae': cv_maes['lgb_sqrt_mae']}, f)
    print(f"  ✅ lgb_sqrt 저장 완료 (CV MAE: {cv_maes['lgb_sqrt_mae']:.4f})", flush=True)
oof_preds['lgb_sqrt_mae'] = m3_oof
test_preds['lgb_sqrt_mae'] = m3_test
print(f"  모델 3 CV MAE: {cv_maes['lgb_sqrt_mae']:.4f}", flush=True)

# 모델 4: XGBoost raw+MAE
print("\n  [모델 4] XGB raw+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase11_xgb.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    m4_oof, m4_test, cv_maes['xgb_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  ⏭️ xgb 캐시 사용 (CV MAE: {cv_maes['xgb_raw_mae']:.4f})", flush=True)
else:
    m4_oof = np.zeros(len(X), dtype=np.float32)
    m4_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        m = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=8,
            min_child_weight=6, subsample=0.9, colsample_bytree=0.85,
            reg_lambda=1.5, reg_alpha=0.05,
            objective='reg:absoluteerror', eval_metric='mae',
            tree_method='hist', random_state=42, verbosity=0, early_stopping_rounds=100)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=combined_weight[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
        pred = np.clip(m.predict(X.iloc[va_idx]), 0, None).astype(np.float32)
        m4_oof[va_idx] = pred
        m4_test += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
    cv_maes['xgb_raw_mae'] = mean_absolute_error(y, m4_oof)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': m4_oof, 'test': m4_test, 'cv_mae': cv_maes['xgb_raw_mae']}, f)
    print(f"  ✅ xgb 저장 완료 (CV MAE: {cv_maes['xgb_raw_mae']:.4f})", flush=True)
oof_preds['xgb_raw_mae'] = m4_oof
test_preds['xgb_raw_mae'] = m4_test
print(f"  모델 4 CV MAE: {cv_maes['xgb_raw_mae']:.4f}", flush=True)

# 모델 5: CatBoost log1p+MAE
print("\n  [모델 5] Cat log1p+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase11_cat_log1p.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    m5_oof, m5_test, cv_maes['cat_log1p_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  ⏭️ cat_log1p 캐시 사용 (CV MAE: {cv_maes['cat_log1p_mae']:.4f})", flush=True)
else:
    m5_oof = np.zeros(len(X), dtype=np.float32)
    m5_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        train_pool = Pool(X.iloc[tr_idx], y_log.iloc[tr_idx], weight=combined_weight[tr_idx])
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
    cv_maes['cat_log1p_mae'] = mean_absolute_error(y, m5_oof)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': m5_oof, 'test': m5_test, 'cv_mae': cv_maes['cat_log1p_mae']}, f)
    print(f"  ✅ cat_log1p 저장 완료 (CV MAE: {cv_maes['cat_log1p_mae']:.4f})", flush=True)
oof_preds['cat_log1p_mae'] = m5_oof
test_preds['cat_log1p_mae'] = m5_test
print(f"  모델 5 CV MAE: {cv_maes['cat_log1p_mae']:.4f}", flush=True)

# 모델 6: CatBoost raw+MAE
print("\n  [모델 6] Cat raw+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase11_cat_raw.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    m6_oof, m6_test, cv_maes['cat_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  ⏭️ cat_raw 캐시 사용 (CV MAE: {cv_maes['cat_raw_mae']:.4f})", flush=True)
else:
    m6_oof = np.zeros(len(X), dtype=np.float32)
    m6_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], weight=combined_weight[tr_idx])
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
    cv_maes['cat_raw_mae'] = mean_absolute_error(y, m6_oof)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': m6_oof, 'test': m6_test, 'cv_mae': cv_maes['cat_raw_mae']}, f)
    print(f"  ✅ cat_raw 저장 완료 (CV MAE: {cv_maes['cat_raw_mae']:.4f})", flush=True)
oof_preds['cat_raw_mae'] = m6_oof
test_preds['cat_raw_mae'] = m6_test
print(f"  모델 6 CV MAE: {cv_maes['cat_raw_mae']:.4f}", flush=True)

# ============================================================
# 5. NN 모델 1: Keras MLP (combined_weight 사용)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== NN 모델: Keras MLP (Adversarial Weight 적용) ===", flush=True)
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

ckpt_path = 'output/ckpt_phase11_mlp.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    mlp_oof, mlp_test, mlp_cv = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  ⏭️ mlp 캐시 사용 (CV MAE: {mlp_cv:.4f})", flush=True)
else:
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
            sample_weight=combined_weight[tr_idx],
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
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': mlp_oof, 'test': mlp_test, 'cv_mae': mlp_cv}, f)
    print(f"  ✅ mlp 저장 완료 (CV MAE: {mlp_cv:.4f})", flush=True)
oof_preds['keras_mlp'] = mlp_oof
test_preds['keras_mlp'] = mlp_test
cv_maes['keras_mlp'] = mlp_cv
print(f"  Keras MLP CV MAE: {mlp_cv:.4f}", flush=True)

# ============================================================
# 6. NN 모델 2: TabNet (sample_weight 미지원, 기존 방식)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== NN 모델: TabNet ===", flush=True)
print("=" * 60, flush=True)

from pytorch_tabnet.tab_model import TabNetRegressor
import torch

ckpt_path = 'output/ckpt_phase11_tabnet.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    tabnet_oof, tabnet_test, tabnet_cv = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  ⏭️ tabnet 캐시 사용 (CV MAE: {tabnet_cv:.4f})", flush=True)
else:
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
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': tabnet_oof, 'test': tabnet_test, 'cv_mae': tabnet_cv}, f)
    print(f"  ✅ tabnet 저장 완료 (CV MAE: {tabnet_cv:.4f})", flush=True)
oof_preds['tabnet'] = tabnet_oof
test_preds['tabnet'] = tabnet_test
cv_maes['tabnet'] = tabnet_cv
print(f"  TabNet CV MAE: {tabnet_cv:.4f}", flush=True)

# ============================================================
# 7. NN 모델 3: Transformer 시계열 모델 (NEW)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== NN 모델: Transformer 시계열 (NEW) ===", flush=True)
print("=" * 60, flush=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- 시계열 데이터 재구성 ---
print("  시계열 시퀀스 구성 중...", flush=True)

# ts_data를 original_idx 순으로 정렬 (train_fe, test_fe 순서와 일치)
ts_data = ts_data.sort_values('_original_idx').reset_index(drop=True)
ts_train = ts_data[ts_data['_is_train'] == 1].copy()
ts_test = ts_data[ts_data['_is_train'] == 0].copy()

# 원본 피처만 추출 + NaN->0 + StandardScaler
ts_scaler = StandardScaler()
ts_train_raw = np.nan_to_num(ts_train[raw_ts_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
ts_test_raw = np.nan_to_num(ts_test[raw_ts_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
ts_train_raw = ts_scaler.fit_transform(ts_train_raw).astype(np.float32)
ts_test_raw = ts_scaler.transform(ts_test_raw).astype(np.float32)

# scenario_id별로 시퀀스 구성 (각 행에 대해 0~timeslot까지 시퀀스, zero-padding)
MAX_SEQ = 25

def build_sequences(scenario_ids, timeslots, features, max_seq=MAX_SEQ):
    """각 행에 대해 해당 scenario의 0~timeslot까지를 입력으로 구성. zero-padding."""
    n_samples = len(scenario_ids)
    n_feat = features.shape[1]
    sequences = np.zeros((n_samples, max_seq, n_feat), dtype=np.float32)
    seq_lengths = np.zeros(n_samples, dtype=np.int32)

    # scenario별로 그룹화
    unique_scenarios = np.unique(scenario_ids)
    scenario_to_rows = {}
    for sc in unique_scenarios:
        mask = scenario_ids == sc
        indices = np.where(mask)[0]
        scenario_to_rows[sc] = indices

    for sc, indices in scenario_to_rows.items():
        sc_timeslots = timeslots[indices]
        sc_features = features[indices]
        # timeslot 순서로 정렬
        sort_order = np.argsort(sc_timeslots)
        sorted_indices = indices[sort_order]
        sorted_features = sc_features[sort_order]

        for local_i, global_i in enumerate(sorted_indices):
            seq_len = local_i + 1  # 0~local_i까지
            sequences[global_i, :seq_len, :] = sorted_features[:seq_len]
            seq_lengths[global_i] = seq_len

    return sequences, seq_lengths

train_scenarios = ts_train['scenario_id'].values
train_timeslots = ts_train['implicit_timeslot'].values.astype(np.int32)
test_scenarios = ts_test['scenario_id'].values
test_timeslots = ts_test['implicit_timeslot'].values.astype(np.int32)

print("  Train 시퀀스 구성...", flush=True)
train_seqs, train_seq_lens = build_sequences(train_scenarios, train_timeslots, ts_train_raw)
print(f"  Train 시퀀스: {train_seqs.shape}", flush=True)

print("  Test 시퀀스 구성...", flush=True)
test_seqs, test_seq_lens = build_sequences(test_scenarios, test_timeslots, ts_test_raw)
print(f"  Test 시퀀스: {test_seqs.shape}", flush=True)

del ts_data, ts_train, ts_test, ts_train_raw, ts_test_raw
gc.collect()

# --- Transformer 모델 정의 ---
class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, MAX_SEQ, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, seq_lens=None):
        # x: (batch, MAX_SEQ, n_features)
        x = self.input_proj(x) + self.pos_emb
        # 패딩 마스크 생성
        if seq_lens is not None:
            batch_size = x.size(0)
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            mask = mask >= seq_lens.unsqueeze(1)  # True = 패딩
            x = self.transformer(x, src_key_padding_mask=mask)
            # 패딩이 아닌 부분만 평균
            valid_mask = (~mask).unsqueeze(-1).float()
            x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            x = self.transformer(x)
            x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

class SeqDataset(Dataset):
    def __init__(self, sequences, seq_lens, targets=None, weights=None):
        self.sequences = torch.FloatTensor(sequences)
        self.seq_lens = torch.IntTensor(seq_lens)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.weights = torch.FloatTensor(weights) if weights is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {'seq': self.sequences[idx], 'seq_len': self.seq_lens[idx]}
        if self.targets is not None:
            item['target'] = self.targets[idx]
        if self.weights is not None:
            item['weight'] = self.weights[idx]
        return item

# --- Transformer 학습 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}", flush=True)

ckpt_path = 'output/ckpt_phase11_transformer.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    transformer_oof, transformer_test, transformer_cv = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  ⏭️ transformer 캐시 사용 (CV MAE: {transformer_cv:.4f})", flush=True)
else:
    transformer_oof = np.zeros(len(X), dtype=np.float32)
    transformer_test = np.zeros(len(X_test), dtype=np.float32)

    # GroupKFold 사용 (scenario_id 기반)
    train_groups = groups.values

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n  Transformer Fold {fold_i+1}/5...", flush=True)
        torch.manual_seed(42 + fold_i)
        np.random.seed(42 + fold_i)

        # 데이터 준비
        tr_dataset = SeqDataset(train_seqs[tr_idx], train_seq_lens[tr_idx],
                                y_log_nn[tr_idx], combined_weight[tr_idx])
        va_dataset = SeqDataset(train_seqs[va_idx], train_seq_lens[va_idx],
                                y_log_nn[va_idx])
        te_dataset = SeqDataset(test_seqs, test_seq_lens)

        tr_loader = DataLoader(tr_dataset, batch_size=256, shuffle=True, num_workers=0)
        va_loader = DataLoader(va_dataset, batch_size=512, shuffle=False, num_workers=0)
        te_loader = DataLoader(te_dataset, batch_size=512, shuffle=False, num_workers=0)

        model = TimeSeriesTransformer(n_features=n_raw_features, d_model=128, nhead=8, num_layers=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(50):
            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0
            for batch in tr_loader:
                seq = batch['seq'].to(device)
                sl = batch['seq_len'].to(device)
                target = batch['target'].to(device)
                weight = batch['weight'].to(device)

                optimizer.zero_grad()
                pred = model(seq, sl)
                loss = (torch.abs(pred - target) * weight).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validate
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch in va_loader:
                    seq = batch['seq'].to(device)
                    sl = batch['seq_len'].to(device)
                    pred = model(seq, sl)
                    val_preds.append(pred.cpu().numpy())
                    val_targets.append(batch['target'].numpy())

            val_preds_arr = np.concatenate(val_preds)
            val_targets_arr = np.concatenate(val_targets)
            # log1p -> raw MAE
            val_raw_pred = np.clip(np.expm1(val_preds_arr), 0, None)
            val_raw_true = np.clip(np.expm1(val_targets_arr), 0, None)
            val_mae = mean_absolute_error(val_raw_true, val_raw_pred)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/50 - Train Loss: {train_loss/n_batches:.4f}, Val MAE: {val_mae:.4f}", flush=True)

            if val_mae < best_val_loss:
                best_val_loss = val_mae
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 8:
                    print(f"    Early stopping at epoch {epoch+1}", flush=True)
                    break

        # Best model로 OOF + Test 예측
        model.load_state_dict(best_state)
        model.eval()

        # OOF
        oof_preds_fold = []
        with torch.no_grad():
            for batch in va_loader:
                seq = batch['seq'].to(device)
                sl = batch['seq_len'].to(device)
                pred = model(seq, sl)
                oof_preds_fold.append(pred.cpu().numpy())
        oof_preds_fold = np.concatenate(oof_preds_fold)
        transformer_oof[va_idx] = np.clip(np.expm1(oof_preds_fold), 0, None).astype(np.float32)

        # Test
        test_preds_fold = []
        with torch.no_grad():
            for batch in te_loader:
                seq = batch['seq'].to(device)
                sl = batch['seq_len'].to(device)
                pred = model(seq, sl)
                test_preds_fold.append(pred.cpu().numpy())
        test_preds_fold = np.concatenate(test_preds_fold)
        transformer_test += np.clip(np.expm1(test_preds_fold), 0, None).astype(np.float32) / 5

        fold_mae = mean_absolute_error(y.values[va_idx], transformer_oof[va_idx])
        print(f"    Fold {fold_i+1} MAE: {fold_mae:.4f} (best val: {best_val_loss:.4f})", flush=True)

        del model, best_state, tr_dataset, va_dataset, te_dataset
        del tr_loader, va_loader, te_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    transformer_cv = mean_absolute_error(y, transformer_oof)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'oof': transformer_oof, 'test': transformer_test, 'cv_mae': transformer_cv}, f)
    print(f"  ✅ transformer 저장 완료 (CV MAE: {transformer_cv:.4f})", flush=True)

oof_preds['transformer'] = transformer_oof
test_preds['transformer'] = transformer_test
cv_maes['transformer'] = transformer_cv
print(f"  Transformer CV MAE: {transformer_cv:.4f}", flush=True)

del train_seqs, test_seqs, train_seq_lens, test_seq_lens
gc.collect()

# ============================================================
# 8. Level 1 가중 평균 앙상블 (9모델)
# ============================================================
print("\n=== Level 1 가중 평균 앙상블 (9모델) ===", flush=True)

model_names = list(oof_preds.keys())
tree_names = [n for n in model_names if n not in ('keras_mlp', 'tabnet', 'transformer')]
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

# 전체 9모델 앙상블
all_w, all_cv = opt_weights(oof_matrix, y)
print("\n  트리+NN 가중치 (9모델):", flush=True)
for n, w in zip(model_names, all_w):
    print(f"    {n:20s}: {w:.4f}", flush=True)
print(f"  트리+NN Level 1 CV MAE (9모델): {all_cv:.4f}", flush=True)

# ============================================================
# 9. Level 2 LGB 스태킹 (9모델)
# ============================================================
print("\n=== Level 2 LGB 스태킹 (9모델) ===", flush=True)

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
    'all_l1_9model': (all_cv, oof_matrix @ all_w, test_matrix @ all_w),
    'lgb_stack_9model': (meta_lgb_cv, meta_lgb_oof, meta_lgb_test),
}

print("\n  비교:", flush=True)
for name, (cv, _, _) in candidates.items():
    print(f"    {name:20s}: {cv:.4f}", flush=True)

best_name = min(candidates, key=lambda k: candidates[k][0])
final_cv = candidates[best_name][0]
final_test = candidates[best_name][2]
print(f"\n  최종 선택: {best_name} (CV MAE: {final_cv:.4f})", flush=True)

# ============================================================
# 10. 결과 비교
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 11 결과 ===", flush=True)
print("=" * 60, flush=True)
print(f"Adversarial AUC: {auc:.4f}", flush=True)
print(f"Combined weight stats: min={combined_weight.min():.2f}, max={combined_weight.max():.2f}, mean={combined_weight.mean():.2f}", flush=True)
print("트리 모델 (adv weight):", flush=True)
for n in tree_names:
    print(f"  {n:20s}: {cv_maes[n]:.4f}", flush=True)
print("NN 모델:", flush=True)
print(f"  {'keras_mlp':20s}: {cv_maes['keras_mlp']:.4f}", flush=True)
print(f"  {'tabnet':20s}: {cv_maes['tabnet']:.4f}", flush=True)
print(f"  {'transformer (NEW)':20s}: {cv_maes['transformer']:.4f}", flush=True)
print(f"Level 1 가중 평균 (트리만):    {tree_cv:.4f}", flush=True)
print(f"Level 1 가중 평균 (9모델):     {all_cv:.4f}", flush=True)
print(f"Level 2 LGB 스태킹 (9모델):   {meta_lgb_cv:.4f}", flush=True)
print(f"최종 선택: {best_name} ({final_cv:.4f})", flush=True)
print(f"Phase 10 대비 개선: {final_cv - 8.577:+.4f}", flush=True)

# ============================================================
# 11. 제출 파일
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
final_test = np.clip(final_test, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase11.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase11.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 12. 피처 중요도 (LGB raw+MAE)
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)

if len(m1_models) > 0:
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': m1_models[0].feature_importances_
    }).sort_values('importance', ascending=False)

    top30 = importance.head(30).sort_values('importance')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top30['feature'], top30['importance'], color='steelblue')
    ax.set_title('Feature Importance Top 30 (Phase 11 - LGB raw+MAE + Adv Weight)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('output/feature_importance_phase11.png', dpi=150, bbox_inches='tight')
    print("feature_importance_phase11.png 저장 완료", flush=True)
else:
    print("  (캐시 사용으로 피처 중요도 시각화 건너뜀)", flush=True)

print("\n=== Phase 11 완료 ===", flush=True)
