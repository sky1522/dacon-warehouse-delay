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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.optimize import minimize
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Google Drive checkpoint helpers
# ============================================================
DRIVE_CKPT_DIR = '/content/drive/MyDrive/dacon_ckpt'
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
print(f"Drive checkpoint dir: {DRIVE_CKPT_DIR}", flush=True)


def save_ckpt(local_path, data):
    with open(local_path, 'wb') as f:
        pickle.dump(data, f)
    drive_path = os.path.join(DRIVE_CKPT_DIR, os.path.basename(local_path))
    with open(drive_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved: {local_path} + {drive_path}", flush=True)


def load_ckpt(local_path):
    drive_path = os.path.join(DRIVE_CKPT_DIR, os.path.basename(local_path))
    if os.path.exists(drive_path):
        with open(drive_path, 'rb') as f:
            print(f"  Drive cache: {drive_path}", flush=True)
            return pickle.load(f)
    if os.path.exists(local_path):
        with open(local_path, 'rb') as f:
            print(f"  Local cache: {local_path}", flush=True)
            return pickle.load(f)
    return None


# ============================================================
# 1. Data Loading
# ============================================================
print("=== Data Load ===", flush=True)
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

N_TRAIN = len(train)

train['_is_train'] = 1
test['_is_train'] = 0
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined['_original_idx'] = range(len(combined))
del train, test
gc.collect()

combined['implicit_timeslot'] = combined.groupby('scenario_id').cumcount()
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)

# --- Time Series Features (64) ---
print("=== Time Series Features ===", flush=True)
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
print("  Time series 64 done", flush=True)

# --- Layout join + basic features ---
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

# --- Phase 3B Interactions (8) ---
combined['orders_per_packstation'] = combined['order_inflow_15m'] / combined['pack_station_count'].replace(0, np.nan)
combined['pack_dock_pressure'] = combined['pack_utilization'] * combined['loading_dock_util']
combined['dock_wait_pressure'] = combined['outbound_truck_wait_min'] * combined['loading_dock_util']
combined['shift_load_pressure'] = combined['prev_shift_volume'] * combined['order_inflow_15m']
combined['battery_congestion'] = combined['low_battery_ratio'] * combined['congestion_score']
combined['storage_density_congestion'] = combined['storage_density_pct'] * combined['congestion_score']
combined['battery_trip_pressure'] = combined['low_battery_ratio'] * combined['avg_trip_distance']
combined['demand_density'] = combined['order_inflow_15m'] * combined['max_zone_density']

# --- Missing indicators ---
train_part = combined[combined['_is_train'] == 1]
missing_counts = train_part.isnull().sum().sort_values(ascending=False)
top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith('_')]
for col in top10_missing:
    if col in combined.columns:
        combined[f'{col}_missing'] = combined[col].isnull().astype(int)
del train_part
gc.collect()

# --- Onset Features (8) ---
print("=== Onset Features ===", flush=True)


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
print("  Onset 8 done", flush=True)

# --- Expanding Mean (30) ---
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

# --- Nonlinear (7) ---
combined['battery_mean_below_44'] = np.maximum(44.0 - combined['battery_mean'], 0).astype(np.float32)
combined['low_battery_ratio_above_02'] = np.maximum(combined['low_battery_ratio'] - 0.2, 0).astype(np.float32)
combined['pack_utilization_sq'] = (combined['pack_utilization'] ** 2).astype(np.float32)
combined['loading_dock_util_sq'] = (combined['loading_dock_util'] ** 2).astype(np.float32)
combined['congestion_score_sq'] = (combined['congestion_score'] ** 2).astype(np.float32)
combined['charge_pressure_nl'] = ((combined['robot_charging'] + combined['charge_queue_length']) / (combined['charger_count'] + 1)).astype(np.float32)
combined['charge_pressure_nl_sq'] = (combined['charge_pressure_nl'] ** 2).astype(np.float32)

# --- Phase (6) ---
combined['is_early_phase'] = (combined['implicit_timeslot'] <= 5).astype(np.float32)
combined['is_mid_phase'] = ((combined['implicit_timeslot'] >= 6) & (combined['implicit_timeslot'] <= 15)).astype(np.float32)
combined['is_late_phase'] = (combined['implicit_timeslot'] >= 16).astype(np.float32)
combined['time_frac'] = (combined['implicit_timeslot'] / 24.0).astype(np.float32)
combined['time_remaining'] = (24 - combined['implicit_timeslot']).astype(np.float32)
combined['time_frac_sq'] = (combined['time_frac'] ** 2).astype(np.float32)

# --- Competitor Features (54) ---
print("=== Competitor Features ===", flush=True)
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

print("  rolling max + deviation...", flush=True)
rollmax_cols = ['order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization', 'loading_dock_util']
for col in rollmax_cols:
    shifted = combined.groupby('scenario_id')[col].shift(1)
    rollmax3 = shifted.groupby(combined['scenario_id']).rolling(3, min_periods=1).max().droplevel(0).sort_index()
    combined[f'{col}_rollmax3_prev'] = rollmax3.astype(np.float32)
    combined[f'{col}_dev_rollmax3'] = (combined[col] - combined[f'{col}_rollmax3_prev']).astype(np.float32)

# --- Expanding Extension B (20) ---
print("=== Expanding Extension (std, max) ===", flush=True)
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
        print(f"  Expanding Extension: {i+1}/{len(expanding_ext_cols)}", flush=True)

print("  Phase 10 features done (319)", flush=True)

# ============================================================
# 2. Queueing Theory Features (42)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Queueing Theory Features (42) ===", flush=True)
print("=" * 60, flush=True)

EPS = 1e-3

print("  A) Utilization / Traffic Intensity (16)...", flush=True)
combined['q_rho_robot'] = (combined['robot_active'] / (combined['robot_total'] + EPS)).clip(0, 0.99).astype(np.float32)
combined['q_rho_charger'] = (combined['robot_charging'] / (combined['charger_count'] + EPS)).clip(0, 0.99).astype(np.float32)
combined['q_rho_pack'] = combined['pack_utilization'].clip(0, 0.99).astype(np.float32)
combined['q_rho_loading'] = combined['loading_dock_util'].clip(0, 0.99).astype(np.float32)

for name in ['robot', 'charger', 'pack', 'loading']:
    rho = combined[f'q_rho_{name}']
    combined[f'q_rho_{name}_sq'] = (rho ** 2).astype(np.float32)
    combined[f'q_rho_{name}_inv'] = (1.0 / (1.0 - rho + EPS)).astype(np.float32)
    combined[f'q_pk_{name}'] = (rho**2 / (1.0 - rho + EPS)).astype(np.float32)

print("  B) Little's Law (6)...", flush=True)
arrival_rate = (combined['order_inflow_15m'] / 15.0).astype(np.float32)
combined['q_arrival_rate'] = arrival_rate
combined['q_expected_charge_wait'] = (combined['charge_queue_length'] / (arrival_rate + EPS)).astype(np.float32)
combined['q_effective_service_robot'] = (combined['robot_active'] / (combined['avg_trip_distance'] + EPS)).astype(np.float32)
combined['q_arrival_service_gap'] = (arrival_rate - combined['q_effective_service_robot']).astype(np.float32)
combined['q_throughput_robot'] = (combined['robot_active'] * (1.0 - combined['congestion_score'])).astype(np.float32)
combined['q_throughput_pack'] = (combined['pack_station_count'] * combined['pack_utilization']).astype(np.float32)

print("  C) Bottleneck Detection (8)...", flush=True)
stages_cols = ['q_rho_robot', 'q_rho_charger', 'q_rho_pack', 'q_rho_loading']
stages_df = combined[stages_cols]
combined['q_bottleneck_max'] = stages_df.max(axis=1).astype(np.float32)
combined['q_bottleneck_min'] = stages_df.min(axis=1).astype(np.float32)
combined['q_bottleneck_mean'] = stages_df.mean(axis=1).astype(np.float32)
combined['q_bottleneck_std'] = stages_df.std(axis=1).astype(np.float32)
combined['q_bottleneck_gap'] = (combined['q_bottleneck_max'] - combined['q_bottleneck_mean']).astype(np.float32)
combined['q_cascade_load_pack'] = (combined['q_rho_loading'] * combined['q_rho_pack']).astype(np.float32)
combined['q_cascade_pack_robot'] = (combined['q_rho_pack'] * combined['q_rho_robot']).astype(np.float32)
combined['q_cascade_all'] = (combined['q_rho_loading'] * combined['q_rho_pack'] * combined['q_rho_robot']).astype(np.float32)

print("  D) Queue Stability Indicators (4)...", flush=True)
combined['q_unstable_robot'] = (combined['q_rho_robot'] > 0.9).astype(np.float32)
combined['q_unstable_charger'] = (combined['q_rho_charger'] > 0.9).astype(np.float32)
combined['q_unstable_pack'] = (combined['q_rho_pack'] > 0.9).astype(np.float32)
combined['q_unstable_count'] = (combined['q_unstable_robot'] + combined['q_unstable_charger'] + combined['q_unstable_pack']).astype(np.float32)

print("  E) Demand Surge / Time-varying (8)...", flush=True)
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)
combined['q_rho_robot_change'] = combined.groupby('scenario_id')['q_rho_robot'].diff().fillna(0).astype(np.float32)
combined['q_rho_robot_accel'] = combined.groupby('scenario_id')['q_rho_robot_change'].diff().fillna(0).astype(np.float32)
combined['q_rho_robot_roll3'] = (
    combined.groupby('scenario_id')['q_rho_robot']
      .rolling(3, min_periods=1).mean()
      .reset_index(level=0, drop=True).astype(np.float32)
)
combined['q_queue_growth'] = combined.groupby('scenario_id')['charge_queue_length'].diff().fillna(0).astype(np.float32)
combined['q_queue_growth_roll3'] = (
    combined.groupby('scenario_id')['q_queue_growth']
      .rolling(3, min_periods=1).mean()
      .reset_index(level=0, drop=True).astype(np.float32)
)
combined['q_inflow_change'] = combined.groupby('scenario_id')['order_inflow_15m'].diff().fillna(0).astype(np.float32)
combined['q_inflow_accel'] = combined.groupby('scenario_id')['q_inflow_change'].diff().fillna(0).astype(np.float32)
combined['q_cum_unstable'] = combined.groupby('scenario_id')['q_unstable_count'].cumsum().astype(np.float32)

q_cols = [c for c in combined.columns if c.startswith('q_')]
for col in q_cols:
    combined[col] = np.nan_to_num(combined[col].values, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
print(f"  Queueing theory features: {len(q_cols)}, cleanup done", flush=True)

# ============================================================
# 3. EDA-based Features (7) - from Phase 13s1
# ============================================================
print("\n=== EDA-based New Features (7) ===", flush=True)

# shift_hour NaN handling
combined['shift_hour'] = combined['shift_hour'].fillna(-1)

combined['is_night_shift'] = (combined['shift_hour'] >= 18).astype('float32')
combined['hours_after_peak'] = (combined['shift_hour'] - 17).clip(lower=0).astype('float32')
combined['shift_phase'] = pd.cut(
    combined['shift_hour'], bins=[-2, 6, 12, 17, 23], labels=[0, 1, 2, 3]
).astype('float32')
# NaN from shift_hour=-1 falling outside bins
combined['shift_phase'] = combined['shift_phase'].fillna(0).astype('float32')

combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)
combined['time_step_in_scenario'] = combined.groupby('scenario_id').cumcount().astype('float32')
combined['ts_squared'] = (combined['time_step_in_scenario'] ** 2).astype('float32')

combined['inflow_per_active_robot'] = (
    combined['order_inflow_15m'] / (combined['robot_active'] + 1.0)
).astype('float32')
combined['is_overload_not_congestion'] = (
    (combined['order_inflow_15m'] > 130) & (combined['congestion_score'] < 8)
).astype('float32')

new_eda_features = ['is_night_shift', 'hours_after_peak', 'shift_phase',
                    'time_step_in_scenario', 'ts_squared',
                    'inflow_per_active_robot', 'is_overload_not_congestion']
for feat in new_eda_features:
    combined[feat] = np.nan_to_num(combined[feat].values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
print("  EDA features: 7 (NaN cleaned)", flush=True)

# ============================================================
# 4. NEW: Bin 9 Rule-based Features (5) - Layer 1
# ============================================================
print("\n=== EDA-based Bin9 Rule Features (5) ===", flush=True)

combined['pack_full'] = (combined['pack_utilization'] >= 0.99).astype('int8')
combined['pack_starved'] = (combined['pack_station_count'] <= 5).astype('int8')
combined['truck_blocked'] = (combined['outbound_truck_wait_min'] >= 25).astype('int8')
combined['pack_critical'] = (combined['pack_full'] * combined['pack_starved']).astype('int8')
combined['high_robot_load'] = (combined['robot_active'] >= 25).astype('int8')

for feat in ['pack_full', 'pack_starved', 'truck_blocked', 'pack_critical', 'high_robot_load']:
    combined[feat] = combined[feat].fillna(0).astype('int8')
    n_pos = combined[feat].sum()
    print(f"  {feat}: {n_pos} positives ({n_pos/len(combined)*100:.2f}%)", flush=True)

# ============================================================
# 5. Drop low-correlation layout features
# ============================================================
print("\n=== Drop Low-Correlation Layout Features ===", flush=True)

LOW_CORR_LAYOUT_COLS = [
    'layout_compactness', 'zone_dispersion', 'one_way_ratio', 'aisle_width_avg',
    'floor_area_sqm', 'ceiling_height_m', 'building_age_years', 'intersection_count',
    'fire_sprinkler_count', 'emergency_exit_count',
]
DERIVED_FROM_DROPPED = [
    'robot_per_area', 'congestion_per_width', 'zone_density_per_width', 'order_per_sqm',
    'warehouse_volume', 'intersection_density', 'pack_station_density', 'charger_density',
    'robot_density_layout', 'movement_friction', 'layout_compact_x_dispersion', 'one_way_friction',
]
all_cols_to_drop = LOW_CORR_LAYOUT_COLS + DERIVED_FROM_DROPPED
cols_to_drop = [c for c in all_cols_to_drop if c in combined.columns]
print(f"Dropping {len(cols_to_drop)} low-corr layout features + derived", flush=True)
combined = combined.drop(columns=cols_to_drop)

# --- float32 + split ---
print("\n=== float32 conversion + split ===", flush=True)
float64_cols = combined.select_dtypes(include='float64').columns
combined[float64_cols] = combined[float64_cols].astype(np.float32)

combined = combined.sort_values('_original_idx').reset_index(drop=True)
train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()

# Keep combined reference for bin9 classifier features (before deleting)
# We need the raw columns for the bin9 classifier
print("  Preparing bin9 classifier features before freeing combined...", flush=True)

BIN9_KS_TOP30 = [
    'pack_utilization', 'pack_station_count', 'outbound_truck_wait_min',
    'robot_active', 'order_inflow_15m', 'robot_utilization',
    'conveyor_speed_mps', 'sku_concentration', 'packaging_material_cost',
    'urgent_order_ratio', 'cold_chain_ratio', 'heavy_item_ratio',
    'avg_items_per_order', 'prev_shift_volume', 'loading_dock_util',
    'inventory_turnover_rate', 'emergency_exit_count', 'staff_on_floor',
    'layout_compactness', 'battery_mean', 'robot_idle', 'unique_sku_15m',
    'forklift_active_count', 'pick_list_length_avg', 'kpi_otd_pct',
    'agv_task_success_rate', 'express_lane_util', 'backorder_ratio',
    'order_wave_count', 'intersection_count'
]
bin9_clf_features = [c for c in BIN9_KS_TOP30 if c in combined.columns]
print(f"  Available features for bin9 classifier: {len(bin9_clf_features)}/{len(BIN9_KS_TOP30)}", flush=True)

X_clf_train = combined.iloc[:N_TRAIN][bin9_clf_features].fillna(0).values.astype('float32')
X_clf_test = combined.iloc[N_TRAIN:][bin9_clf_features].fillna(0).values.astype('float32')

# Now we can safely wait to add OOF features after CV definition
# But we need layout_id and scenario_id which are still in train_fe/test_fe

del combined
gc.collect()

drop_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']
feature_cols_base = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]

y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y)
y_sqrt = np.sqrt(y)
groups = train_fe['layout_id']
time_idx = train_fe['implicit_timeslot'].values.astype(np.float32)

assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID order mismatch!"
print("ID order verified!", flush=True)

# ============================================================
# CV: StratifiedGroupKFold on layout_id
# ============================================================
print("\n=== CV Strategy: StratifiedGroupKFold(layout_id) ===", flush=True)

y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(sgkf.split(train_fe, y_binned, groups=groups))

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    tr_layouts = set(train_fe.iloc[tr_idx]['layout_id'].unique())
    va_layouts = set(train_fe.iloc[va_idx]['layout_id'].unique())
    assert len(tr_layouts & va_layouts) == 0, f"Fold {fold_i+1}: layout overlap!"
    print(f"  Fold {fold_i+1}: train={len(tr_idx):,} val={len(va_idx):,} | overlap=0", flush=True)
print("  CV validation passed!", flush=True)

# ============================================================
# 6. NEW: OOF Bin 9 Probability Feature (Layer 2)
# ============================================================
print("\n=== OOF Bin 9 Probability Feature ===", flush=True)

y_train_vals = y.values
y_bin9 = (y_train_vals > 100).astype(int)
print(f"  Bin 9 train ratio: {y_bin9.mean()*100:.2f}%", flush=True)

oof_bin9_proba = np.zeros(len(y_train_vals), dtype='float32')

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=8,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1, feature_fraction=0.8,
        verbosity=-1, class_weight='balanced', random_state=42,
    )
    clf.fit(X_clf_train[tr_idx], y_bin9[tr_idx])
    oof_bin9_proba[va_idx] = clf.predict_proba(X_clf_train[va_idx])[:, 1].astype('float32')
    print(f"    Fold {fold_idx+1} bin9 classifier done", flush=True)

auc = roc_auc_score(y_bin9, oof_bin9_proba)
print(f"  OOF bin9 classifier AUC: {auc:.4f}", flush=True)
print(f"  OOF proba: mean={oof_bin9_proba.mean():.4f}, max={oof_bin9_proba.max():.4f}, >0.5: {(oof_bin9_proba>0.5).sum()}", flush=True)
assert auc > 0.85, f"Bin9 classifier AUC too low: {auc}"

# Test: full train classifier
print("  Training final bin9 classifier on full train...", flush=True)
clf_final = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=8,
    min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1, feature_fraction=0.8,
    verbosity=-1, class_weight='balanced', random_state=42,
)
clf_final.fit(X_clf_train, y_bin9)
test_bin9_proba = clf_final.predict_proba(X_clf_test)[:, 1].astype('float32')
print(f"  Test proba: mean={test_bin9_proba.mean():.4f}, max={test_bin9_proba.max():.4f}, >0.5: {(test_bin9_proba>0.5).sum()}", flush=True)

# Add to train_fe / test_fe
train_fe['oof_bin9_proba'] = oof_bin9_proba
test_fe['oof_bin9_proba'] = test_bin9_proba

del X_clf_train, X_clf_test
gc.collect()

# ============================================================
# 7. NEW: Layout-level Bin 9 Rate Encoding (Layer 3)
# ============================================================
print("\n=== OOF Layout Bin9 Rate Encoding ===", flush=True)

global_bin9_rate = y_bin9.mean()
oof_layout_bin9_rate = np.zeros(len(y_train_vals), dtype='float32')

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    tr_df_temp = pd.DataFrame({
        'layout_id': train_fe.iloc[tr_idx]['layout_id'].values,
        'is_bin9': y_bin9[tr_idx]
    })
    layout_rate_map = tr_df_temp.groupby('layout_id')['is_bin9'].mean()

    va_layout_ids = train_fe.iloc[va_idx]['layout_id'].values
    oof_layout_bin9_rate[va_idx] = pd.Series(va_layout_ids).map(layout_rate_map).fillna(global_bin9_rate).values

# Test: full train layout rate
train_layout_rate = pd.DataFrame({
    'layout_id': train_fe['layout_id'].values,
    'is_bin9': y_bin9
}).groupby('layout_id')['is_bin9'].mean()

test_layout_ids = test_fe['layout_id'].values
test_layout_bin9_rate = pd.Series(test_layout_ids).map(train_layout_rate).fillna(global_bin9_rate).values.astype('float32')

print(f"  Train OOF layout bin9 rate: mean={oof_layout_bin9_rate.mean():.4f}, max={oof_layout_bin9_rate.max():.4f}", flush=True)
print(f"  Test layout bin9 rate: mean={test_layout_bin9_rate.mean():.4f}, max={test_layout_bin9_rate.max():.4f}", flush=True)
n_mapped = (~np.isnan(pd.Series(test_layout_ids).map(train_layout_rate).values)).sum()
print(f"  Test layouts mapped from train: {n_mapped}/{len(test_layout_ids)}", flush=True)

train_fe['layout_bin9_rate'] = oof_layout_bin9_rate
test_fe['layout_bin9_rate'] = test_layout_bin9_rate

# Verify all 7 new bin9 features
print("\n=== New 7 bin9 features summary ===", flush=True)
new_bin9_features = ['pack_full', 'pack_starved', 'truck_blocked', 'pack_critical',
                     'high_robot_load', 'oof_bin9_proba', 'layout_bin9_rate']
for feat in new_bin9_features:
    n_nan_tr = train_fe[feat].isna().sum()
    n_nan_te = test_fe[feat].isna().sum()
    print(f"  {feat}: train_nan={n_nan_tr}, test_nan={n_nan_te}, "
          f"train_mean={train_fe[feat].mean():.4f}, test_mean={test_fe[feat].mean():.4f}", flush=True)
    assert n_nan_tr == 0, f"NaN in train {feat}"
    assert n_nan_te == 0, f"NaN in test {feat}"

# ============================================================
# Update feature_cols with new features
# ============================================================
feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
print(f"\nTotal features: {len(feature_cols)} (base {len(feature_cols_base)} + 7 bin9)", flush=True)

X = train_fe[feature_cols]
X_test = test_fe[feature_cols]

# --- Sample weight (base only, no tail weight) ---
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
print(f"  Weights: min={sample_w.min():.2f}, max={sample_w.max():.2f}, mean={sample_w.mean():.2f}", flush=True)

# ============================================================
# NN Preprocessing
# ============================================================
print("\n=== NN Preprocessing ===", flush=True)
scaler = StandardScaler()
X_train_nn = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_test_nn = np.nan_to_num(X_test.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_train_nn = scaler.fit_transform(X_train_nn).astype(np.float32)
X_test_nn = scaler.transform(X_test_nn).astype(np.float32)
y_log_nn = y_log.values.astype(np.float32)
print(f"  NN data: train {X_train_nn.shape}, test {X_test_nn.shape}", flush=True)

# ============================================================
# Tree 6 Models (Phase 13s1 identical)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Tree 6 Models ===", flush=True)
print("=" * 60, flush=True)

oof_preds = {}
test_preds = {}
cv_maes = {}

# Model 1: LGB raw+MAE
print("\n  [Model 1] LGB raw+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase13s5a_lgb_raw.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m1_oof, m1_test, cv_maes['lgb_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    m1_models = []
    print(f"  Cache hit (CV MAE: {cv_maes['lgb_raw_mae']:.4f})", flush=True)
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
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sample_w[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred = np.clip(m.predict(X.iloc[va_idx]), 0, None).astype(np.float32)
        m1_oof[va_idx] = pred
        m1_test += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
        m1_models.append(m)
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
    cv_maes['lgb_raw_mae'] = mean_absolute_error(y, m1_oof)
    save_ckpt(ckpt_path, {'oof': m1_oof, 'test': m1_test, 'cv_mae': cv_maes['lgb_raw_mae']})
oof_preds['lgb_raw_mae'] = m1_oof
test_preds['lgb_raw_mae'] = m1_test
print(f"  Model 1 CV MAE: {cv_maes['lgb_raw_mae']:.4f}", flush=True)

# Model 2: LGB log1p+Huber
print("\n  [Model 2] LGB log1p+Huber...", flush=True)
ckpt_path = 'output/ckpt_phase13s5a_lgb_huber.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m2_oof, m2_test, cv_maes['lgb_log1p_huber'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
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
    cv_maes['lgb_log1p_huber'] = mean_absolute_error(y, m2_oof)
    save_ckpt(ckpt_path, {'oof': m2_oof, 'test': m2_test, 'cv_mae': cv_maes['lgb_log1p_huber']})
oof_preds['lgb_log1p_huber'] = m2_oof
test_preds['lgb_log1p_huber'] = m2_test
print(f"  Model 2 CV MAE: {cv_maes['lgb_log1p_huber']:.4f}", flush=True)

# Model 3: LGB sqrt+MAE
print("\n  [Model 3] LGB sqrt+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase13s5a_lgb_sqrt.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m3_oof, m3_test, cv_maes['lgb_sqrt_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
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
    cv_maes['lgb_sqrt_mae'] = mean_absolute_error(y, m3_oof)
    save_ckpt(ckpt_path, {'oof': m3_oof, 'test': m3_test, 'cv_mae': cv_maes['lgb_sqrt_mae']})
oof_preds['lgb_sqrt_mae'] = m3_oof
test_preds['lgb_sqrt_mae'] = m3_test
print(f"  Model 3 CV MAE: {cv_maes['lgb_sqrt_mae']:.4f}", flush=True)

# Model 4: XGB raw+MAE
print("\n  [Model 4] XGB raw+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase13s5a_xgb.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m4_oof, m4_test, cv_maes['xgb_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
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
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sample_w[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
        pred = np.clip(m.predict(X.iloc[va_idx]), 0, None).astype(np.float32)
        m4_oof[va_idx] = pred
        m4_test += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.iloc[va_idx], pred):.4f}", flush=True)
    cv_maes['xgb_raw_mae'] = mean_absolute_error(y, m4_oof)
    save_ckpt(ckpt_path, {'oof': m4_oof, 'test': m4_test, 'cv_mae': cv_maes['xgb_raw_mae']})
oof_preds['xgb_raw_mae'] = m4_oof
test_preds['xgb_raw_mae'] = m4_test
print(f"  Model 4 CV MAE: {cv_maes['xgb_raw_mae']:.4f}", flush=True)

# Model 5: CatBoost log1p+MAE
print("\n  [Model 5] Cat log1p+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase13s5a_cat_log1p.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m5_oof, m5_test, cv_maes['cat_log1p_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
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
    cv_maes['cat_log1p_mae'] = mean_absolute_error(y, m5_oof)
    save_ckpt(ckpt_path, {'oof': m5_oof, 'test': m5_test, 'cv_mae': cv_maes['cat_log1p_mae']})
oof_preds['cat_log1p_mae'] = m5_oof
test_preds['cat_log1p_mae'] = m5_test
print(f"  Model 5 CV MAE: {cv_maes['cat_log1p_mae']:.4f}", flush=True)

# Model 6: CatBoost raw+MAE
print("\n  [Model 6] Cat raw+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase13s5a_cat_raw.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m6_oof, m6_test, cv_maes['cat_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
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
    cv_maes['cat_raw_mae'] = mean_absolute_error(y, m6_oof)
    save_ckpt(ckpt_path, {'oof': m6_oof, 'test': m6_test, 'cv_mae': cv_maes['cat_raw_mae']})
oof_preds['cat_raw_mae'] = m6_oof
test_preds['cat_raw_mae'] = m6_test
print(f"  Model 6 CV MAE: {cv_maes['cat_raw_mae']:.4f}", flush=True)

# ============================================================
# NN Model 1: Keras MLP
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== NN Model: Keras MLP ===", flush=True)
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

ckpt_path = 'output/ckpt_phase13s5a_mlp.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    mlp_oof, mlp_test, mlp_cv = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  Cache hit (CV MAE: {mlp_cv:.4f})", flush=True)
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
        model.fit(X_train_nn[tr_idx], y_log_nn[tr_idx],
                  validation_data=(X_train_nn[va_idx], y_log_nn[va_idx]),
                  epochs=100, batch_size=512, callbacks=[es, rlr], verbose=0)
        pred_log = model.predict(X_train_nn[va_idx], verbose=0).flatten()
        pred = np.clip(np.expm1(pred_log), 0, None).astype(np.float32)
        mlp_oof[va_idx] = pred
        mlp_test += np.clip(np.expm1(model.predict(X_test_nn, verbose=0).flatten()), 0, None).astype(np.float32) / 5
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.values[va_idx], pred):.4f}", flush=True)
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    mlp_cv = mean_absolute_error(y, mlp_oof)
    save_ckpt(ckpt_path, {'oof': mlp_oof, 'test': mlp_test, 'cv_mae': mlp_cv})
oof_preds['keras_mlp'] = mlp_oof
test_preds['keras_mlp'] = mlp_test
cv_maes['keras_mlp'] = mlp_cv
print(f"  Keras MLP CV MAE: {mlp_cv:.4f}", flush=True)

# ============================================================
# NN Model 2: TabNet
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== NN Model: TabNet ===", flush=True)
print("=" * 60, flush=True)

from pytorch_tabnet.tab_model import TabNetRegressor
import torch

ckpt_path = 'output/ckpt_phase13s5a_tabnet.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    tabnet_oof, tabnet_test, tabnet_cv = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    print(f"  Cache hit (CV MAE: {tabnet_cv:.4f})", flush=True)
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
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax', seed=42, verbose=10,
        )
        model.fit(X_train_nn[tr_idx], y_log_nn[tr_idx].reshape(-1, 1),
                  eval_set=[(X_train_nn[va_idx], y_log_nn[va_idx].reshape(-1, 1))],
                  eval_metric=['mae'], max_epochs=100, patience=15,
                  batch_size=2048, virtual_batch_size=256)
        pred_log = model.predict(X_train_nn[va_idx]).flatten()
        pred = np.clip(np.expm1(pred_log), 0, None).astype(np.float32)
        tabnet_oof[va_idx] = pred
        tabnet_test += np.clip(np.expm1(model.predict(X_test_nn).flatten()), 0, None).astype(np.float32) / 5
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.values[va_idx], pred):.4f}", flush=True)
        del model
        gc.collect()
    tabnet_cv = mean_absolute_error(y, tabnet_oof)
    save_ckpt(ckpt_path, {'oof': tabnet_oof, 'test': tabnet_test, 'cv_mae': tabnet_cv})
oof_preds['tabnet'] = tabnet_oof
test_preds['tabnet'] = tabnet_test
cv_maes['tabnet'] = tabnet_cv
print(f"  TabNet CV MAE: {tabnet_cv:.4f}", flush=True)

# ============================================================
# Level 1 Weighted Average Ensemble
# ============================================================
print("\n=== Level 1 Weighted Average Ensemble ===", flush=True)

model_names = list(oof_preds.keys())
tree_names = [n for n in model_names if n not in ('keras_mlp', 'tabnet')]
oof_matrix = np.column_stack([oof_preds[n] for n in model_names])
test_matrix = np.column_stack([test_preds[n] for n in model_names])
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

tree_w, tree_cv = opt_weights(tree_oof_matrix, y)
print("  Tree only weights:", flush=True)
for n, w in zip(tree_names, tree_w):
    print(f"    {n:20s}: {w:.4f}", flush=True)
print(f"  Tree only Level 1 CV MAE: {tree_cv:.4f}", flush=True)

all_w, all_cv = opt_weights(oof_matrix, y)
print("\n  Tree+NN weights:", flush=True)
for n, w in zip(model_names, all_w):
    print(f"    {n:20s}: {w:.4f}", flush=True)
print(f"  Tree+NN Level 1 CV MAE: {all_cv:.4f}", flush=True)

# ============================================================
# Level 2 LGB Stacking
# ============================================================
print("\n=== Level 2 LGB Stacking (8 models) ===", flush=True)

meta_cols_extra = ['implicit_timeslot', 'order_inflow_15m', 'battery_mean',
                   'robot_active', 'pack_utilization', 'congestion_score']
meta_train_extra = train_fe[meta_cols_extra].values.astype(np.float32)
meta_test_extra = test_fe[meta_cols_extra].values.astype(np.float32)

meta_X_train = np.nan_to_num(np.column_stack([oof_matrix, meta_train_extra]), nan=0.0)
meta_X_test = np.nan_to_num(np.column_stack([test_matrix, meta_test_extra]), nan=0.0)
meta_feature_names = model_names + meta_cols_extra
print(f"  Meta features: {len(meta_feature_names)}", flush=True)

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
print(f"  Level 2 LGB Stacking CV MAE: {meta_lgb_cv:.4f}", flush=True)

# --- Final selection ---
candidates = {
    'tree_only_l1': (tree_cv, tree_oof_matrix @ tree_w, tree_test_matrix @ tree_w),
    'all_l1': (all_cv, oof_matrix @ all_w, test_matrix @ all_w),
    'lgb_stack': (meta_lgb_cv, meta_lgb_oof, meta_lgb_test),
}
print("\n  Comparison:", flush=True)
for name, (cv, _, _) in candidates.items():
    print(f"    {name:20s}: {cv:.4f}", flush=True)

best_name = min(candidates, key=lambda k: candidates[k][0])
final_cv = candidates[best_name][0]
final_oof = candidates[best_name][1]
final_test = candidates[best_name][2]
print(f"\n  Final selection: {best_name} (CV MAE: {final_cv:.4f})", flush=True)

# ============================================================
# Hard Layout Residual Analysis
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Hard Layout Residual Analysis ===", flush=True)
print("=" * 60, flush=True)

hard_ranking_path = 'output/phase13s2_analysis/layout_mae_ranking.csv'
if os.path.exists(hard_ranking_path):
    layout_ranking = pd.read_csv(hard_ranking_path)
    if 'difficulty' in layout_ranking.columns:
        hard_ids = layout_ranking[layout_ranking['difficulty'] == 'hard']['layout_id'].tolist()
    else:
        hard_threshold = layout_ranking['mae'].quantile(0.8)
        hard_ids = layout_ranking[layout_ranking['mae'] > hard_threshold]['layout_id'].tolist()
else:
    print("  WARNING: layout_mae_ranking.csv not found", flush=True)
    hard_ids = []

if hard_ids:
    hard_mask = train_fe['layout_id'].isin(hard_ids).values
    hard_oof_vals = final_oof[hard_mask]
    hard_y_vals = y.values[hard_mask]
    hard_residual = hard_y_vals - hard_oof_vals

    hard_df = pd.DataFrame({'y': hard_y_vals, 'oof': hard_oof_vals, 'residual': hard_residual})
    hard_df['target_bin'] = pd.qcut(hard_df['y'], q=10, labels=False, duplicates='drop')

    bin_stats = hard_df.groupby('target_bin').agg(
        target_mean=('y', 'mean'),
        oof_mean=('oof', 'mean'),
        residual_mean=('residual', 'mean'),
        abs_residual=('residual', lambda x: x.abs().mean()),
        n=('residual', 'size'),
    )
    print(bin_stats.to_string(), flush=True)
    print(f"\nHard MAE: {np.abs(hard_residual).mean():.3f}", flush=True)
    print(f"Phase 13s1 baseline: 18.784", flush=True)
    print(f"Phase 13s3 baseline: 18.471", flush=True)
    if 9 in bin_stats.index:
        print(f"\nbin 9 residual: {bin_stats.loc[9, 'residual_mean']:.2f}", flush=True)
        print(f"  Phase 13s1 baseline: +95.33", flush=True)
        print(f"  Phase 13s3 baseline: +95.85", flush=True)
        print(f"  Goal: < +80", flush=True)

    # All bin 9 samples
    all_bin9_mask = (y.values > 100)
    all_bin9_residual = y.values[all_bin9_mask] - final_oof[all_bin9_mask]
    print(f"\n=== All bin 9 samples (target > 100) ===", flush=True)
    print(f"  count: {all_bin9_mask.sum()}", flush=True)
    print(f"  target mean: {y.values[all_bin9_mask].mean():.2f}", flush=True)
    print(f"  oof mean: {final_oof[all_bin9_mask].mean():.2f}", flush=True)
    print(f"  residual mean: {all_bin9_residual.mean():.2f}", flush=True)
    print(f"  Phase 13s1 baseline: +118.0", flush=True)

# ============================================================
# Results Summary
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 13 Step 5A Results ===", flush=True)
print("=" * 60, flush=True)
print(f"Configuration:", flush=True)
print(f"  CV: StratifiedGroupKFold(layout_id, target_bin=5)", flush=True)
print(f"  Total features: {len(feature_cols)} (346 base + 7 bin9)", flush=True)
print(f"  Sample weight: base only (no tail weight)", flush=True)
print(f"  Bin9 classifier AUC: {auc:.4f}", flush=True)
print("Tree models:", flush=True)
for n in tree_names:
    print(f"  {n:20s}: {cv_maes[n]:.4f}", flush=True)
print("NN models:", flush=True)
print(f"  {'keras_mlp':20s}: {cv_maes['keras_mlp']:.4f}", flush=True)
print(f"  {'tabnet':20s}: {cv_maes['tabnet']:.4f}", flush=True)
print(f"Level 1 tree only:           {tree_cv:.4f}", flush=True)
print(f"Level 1 tree+NN:             {all_cv:.4f}", flush=True)
print(f"Level 2 LGB stacking:        {meta_lgb_cv:.4f}", flush=True)
print(f"Final selection:              {best_name} ({final_cv:.4f})", flush=True)
print(f"\n=== Comparison ===", flush=True)
print(f"Phase 13s1 CV:     8.5668", flush=True)
print(f"Phase 13s3 CV:     8.5766 (tail weight, worse)", flush=True)
print(f"Phase 13s5a CV:    {final_cv:.4f}", flush=True)
print(f"Phase 13s1 Public: 10.0078", flush=True)

# ============================================================
# Submission File
# ============================================================
print("\n=== Submission File ===", flush=True)
final_test = np.clip(final_test, 0, None)
submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase13s5a.csv', index=False)

drive_sub_path = os.path.join(DRIVE_CKPT_DIR, 'submission_phase13s5a.csv')
submission.to_csv(drive_sub_path, index=False)
print(f"  Drive copy: {drive_sub_path}", flush=True)

assert list(submission.columns) == list(sample_sub.columns), "Column mismatch!"
assert len(submission) == len(sample_sub), "Row count mismatch!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID order mismatch!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "Negative predictions!"
print("submission_phase13s5a.csv done", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# Feature Importance
# ============================================================
print("\n=== Feature Importance ===", flush=True)

all_new_features = new_eda_features + new_bin9_features

if len(m1_models) > 0:
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': m1_models[0].feature_importances_
    }).sort_values('importance', ascending=False)

    top30 = importance.head(30).sort_values('importance')
    colors = []
    for f in top30['feature']:
        if f in new_bin9_features:
            colors.append('crimson')
        elif f.startswith('q_'):
            colors.append('orangered')
        elif f in new_eda_features:
            colors.append('forestgreen')
        else:
            colors.append('steelblue')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top30['feature'], top30['importance'], color=colors)
    ax.set_title('Feature Importance Top 30 (Phase 13s5a - LGB raw+MAE)')
    ax.set_xlabel('Importance')
    legend_elements = [Patch(facecolor='steelblue', label='Existing'),
                       Patch(facecolor='orangered', label='Queueing (q_)'),
                       Patch(facecolor='forestgreen', label='EDA Features'),
                       Patch(facecolor='crimson', label='Bin9 Features')]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig('output/feature_importance_phase13s5a.png', dpi=150, bbox_inches='tight')
    print("feature_importance_phase13s5a.png saved", flush=True)

    # New bin9 features ranking
    print(f"\n  New bin9 features importance ranking:", flush=True)
    for feat in new_bin9_features:
        if feat in importance['feature'].values:
            rank = importance[importance['feature'] == feat].index[0]
            rank_pos = (importance['importance'] >= importance.loc[rank, 'importance']).sum()
            imp_val = importance.loc[rank, 'importance']
            print(f"    #{rank_pos:3d}/{len(feature_cols)}: {feat:25s} = {imp_val:.0f}", flush=True)
else:
    print("  (No model objects cached - skipping importance plot)", flush=True)

print("\n=== Phase 13 Step 5A Complete ===", flush=True)
