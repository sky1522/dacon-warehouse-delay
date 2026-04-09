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
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
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
# Part 1: Phase 13s1 base features (346)
# ============================================================
print("=" * 60, flush=True)
print("=== Part 1: Phase 13s1 Base Features ===", flush=True)
print("=" * 60, flush=True)

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

# --- Queueing Theory (42) ---
print("\n=== Queueing Theory Features (42) ===", flush=True)
EPS = 1e-3
combined['q_rho_robot'] = (combined['robot_active'] / (combined['robot_total'] + EPS)).clip(0, 0.99).astype(np.float32)
combined['q_rho_charger'] = (combined['robot_charging'] / (combined['charger_count'] + EPS)).clip(0, 0.99).astype(np.float32)
combined['q_rho_pack'] = combined['pack_utilization'].clip(0, 0.99).astype(np.float32)
combined['q_rho_loading'] = combined['loading_dock_util'].clip(0, 0.99).astype(np.float32)
for name in ['robot', 'charger', 'pack', 'loading']:
    rho = combined[f'q_rho_{name}']
    combined[f'q_rho_{name}_sq'] = (rho ** 2).astype(np.float32)
    combined[f'q_rho_{name}_inv'] = (1.0 / (1.0 - rho + EPS)).astype(np.float32)
    combined[f'q_pk_{name}'] = (rho**2 / (1.0 - rho + EPS)).astype(np.float32)
arrival_rate = (combined['order_inflow_15m'] / 15.0).astype(np.float32)
combined['q_arrival_rate'] = arrival_rate
combined['q_expected_charge_wait'] = (combined['charge_queue_length'] / (arrival_rate + EPS)).astype(np.float32)
combined['q_effective_service_robot'] = (combined['robot_active'] / (combined['avg_trip_distance'] + EPS)).astype(np.float32)
combined['q_arrival_service_gap'] = (arrival_rate - combined['q_effective_service_robot']).astype(np.float32)
combined['q_throughput_robot'] = (combined['robot_active'] * (1.0 - combined['congestion_score'])).astype(np.float32)
combined['q_throughput_pack'] = (combined['pack_station_count'] * combined['pack_utilization']).astype(np.float32)
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
combined['q_unstable_robot'] = (combined['q_rho_robot'] > 0.9).astype(np.float32)
combined['q_unstable_charger'] = (combined['q_rho_charger'] > 0.9).astype(np.float32)
combined['q_unstable_pack'] = (combined['q_rho_pack'] > 0.9).astype(np.float32)
combined['q_unstable_count'] = (combined['q_unstable_robot'] + combined['q_unstable_charger'] + combined['q_unstable_pack']).astype(np.float32)
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)
combined['q_rho_robot_change'] = combined.groupby('scenario_id')['q_rho_robot'].diff().fillna(0).astype(np.float32)
combined['q_rho_robot_accel'] = combined.groupby('scenario_id')['q_rho_robot_change'].diff().fillna(0).astype(np.float32)
combined['q_rho_robot_roll3'] = combined.groupby('scenario_id')['q_rho_robot'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).astype(np.float32)
combined['q_queue_growth'] = combined.groupby('scenario_id')['charge_queue_length'].diff().fillna(0).astype(np.float32)
combined['q_queue_growth_roll3'] = combined.groupby('scenario_id')['q_queue_growth'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).astype(np.float32)
combined['q_inflow_change'] = combined.groupby('scenario_id')['order_inflow_15m'].diff().fillna(0).astype(np.float32)
combined['q_inflow_accel'] = combined.groupby('scenario_id')['q_inflow_change'].diff().fillna(0).astype(np.float32)
combined['q_cum_unstable'] = combined.groupby('scenario_id')['q_unstable_count'].cumsum().astype(np.float32)
q_cols = [c for c in combined.columns if c.startswith('q_')]
for col in q_cols:
    combined[col] = np.nan_to_num(combined[col].values, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
print(f"  Queueing theory: {len(q_cols)} features", flush=True)

# --- EDA Features (7) ---
print("=== EDA Features (7) ===", flush=True)
combined['shift_hour'] = combined['shift_hour'].fillna(-1)
combined['is_night_shift'] = (combined['shift_hour'] >= 18).astype('float32')
combined['hours_after_peak'] = (combined['shift_hour'] - 17).clip(lower=0).astype('float32')
combined['shift_phase'] = pd.cut(combined['shift_hour'], bins=[-2, 6, 12, 17, 23], labels=[0, 1, 2, 3]).astype('float32')
combined['shift_phase'] = combined['shift_phase'].fillna(0).astype('float32')
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)
combined['time_step_in_scenario'] = combined.groupby('scenario_id').cumcount().astype('float32')
combined['ts_squared'] = (combined['time_step_in_scenario'] ** 2).astype('float32')
combined['inflow_per_active_robot'] = (combined['order_inflow_15m'] / (combined['robot_active'] + 1.0)).astype('float32')
combined['is_overload_not_congestion'] = ((combined['order_inflow_15m'] > 130) & (combined['congestion_score'] < 8)).astype('float32')
new_eda_features = ['is_night_shift', 'hours_after_peak', 'shift_phase', 'time_step_in_scenario', 'ts_squared', 'inflow_per_active_robot', 'is_overload_not_congestion']
for feat in new_eda_features:
    combined[feat] = np.nan_to_num(combined[feat].values, nan=0.0).astype(np.float32)

# --- Drop low-corr layout cols ---
print("=== Drop Low-Corr Layout ===", flush=True)
LOW_CORR_LAYOUT_COLS = ['layout_compactness', 'zone_dispersion', 'one_way_ratio', 'aisle_width_avg',
                         'floor_area_sqm', 'ceiling_height_m', 'building_age_years', 'intersection_count',
                         'fire_sprinkler_count', 'emergency_exit_count']
DERIVED_FROM_DROPPED = ['robot_per_area', 'congestion_per_width', 'zone_density_per_width', 'order_per_sqm',
                         'warehouse_volume', 'intersection_density', 'pack_station_density', 'charger_density',
                         'robot_density_layout', 'movement_friction', 'layout_compact_x_dispersion', 'one_way_friction']
cols_to_drop = [c for c in LOW_CORR_LAYOUT_COLS + DERIVED_FROM_DROPPED if c in combined.columns]
combined = combined.drop(columns=cols_to_drop)
print(f"  Dropped {len(cols_to_drop)} columns", flush=True)

# Mark base feature columns (before adding new agg features)
drop_cols_meta = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']
base_feature_cols = [c for c in combined.columns if c not in drop_cols_meta
                     and combined[c].dtype in [np.float32, np.float64, np.int32, np.int64, np.int8]]
print(f"  Base features (Phase 13s1): {len(base_feature_cols)}", flush=True)

# ============================================================
# Part 2: Large-scale Aggregation Features
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Part 2: Large-scale Aggregation Features ===", flush=True)
print("=" * 60, flush=True)

# --- A) Scenario-level aggregation ---
print("\n=== A) Scenario-level aggregation ===", flush=True)

SCENARIO_AGG_COLS = [
    'congestion_score', 'max_zone_density', 'robot_charging',
    'low_battery_ratio', 'robot_idle', 'order_inflow_15m',
    'battery_mean', 'charge_queue_length', 'avg_charge_wait',
    'near_collision_15m', 'blocked_path_15m', 'unique_sku_15m',
    'fault_count_15m', 'avg_recovery_time', 'sku_concentration',
    'urgent_order_ratio', 'battery_std', 'robot_utilization',
    'robot_active', 'pack_utilization', 'loading_dock_util',
    'heavy_item_ratio', 'manual_override_ratio', 'cold_chain_ratio',
    'outbound_truck_wait_min', 'packaging_material_cost',
]
SCENARIO_AGG_COLS = [c for c in SCENARIO_AGG_COLS if c in combined.columns]
print(f"  Scenario agg columns: {len(SCENARIO_AGG_COLS)}", flush=True)

AGG_FUNCS_SCENARIO = ['mean', 'std', 'max', 'min', 'median', 'skew']
for col in SCENARIO_AGG_COLS:
    for func in AGG_FUNCS_SCENARIO:
        new_col = f's_{col}_{func}'
        if func == 'skew':
            combined[new_col] = combined.groupby('scenario_id')[col].transform(
                lambda x: x.skew() if len(x) > 2 else 0
            ).astype('float32')
        else:
            combined[new_col] = combined.groupby('scenario_id')[col].transform(func).astype('float32')

# Scenario first/last/trend (top 15)
for col in SCENARIO_AGG_COLS[:15]:
    combined[f's_{col}_first'] = combined.groupby('scenario_id')[col].transform('first').astype('float32')
    combined[f's_{col}_last'] = combined.groupby('scenario_id')[col].transform('last').astype('float32')
    combined[f's_{col}_trend'] = (combined[f's_{col}_last'] - combined[f's_{col}_first']).astype('float32')

# Distance from max + rank in scenario (top 15)
for col in SCENARIO_AGG_COLS[:15]:
    combined[f's_{col}_dist_from_max'] = (combined[f's_{col}_max'] - combined[col]).astype('float32')
    combined[f's_{col}_rank_in_scenario'] = combined.groupby('scenario_id')[col].rank(pct=True).astype('float32')

n_scenario_feats = len([c for c in combined.columns if c.startswith('s_')])
print(f"  Scenario-level features: {n_scenario_feats}", flush=True)

# --- B) Layout-level aggregation ---
print("\n=== B) Layout-level aggregation ===", flush=True)

LAYOUT_AGG_COLS = [
    'congestion_score', 'robot_active', 'robot_charging',
    'order_inflow_15m', 'pack_utilization', 'charge_queue_length',
    'urgent_order_ratio', 'max_zone_density', 'battery_mean',
    'robot_utilization', 'sku_concentration',
]
LAYOUT_AGG_COLS = [c for c in LAYOUT_AGG_COLS if c in combined.columns]

for col in LAYOUT_AGG_COLS:
    combined[f'l_{col}_mean'] = combined.groupby('layout_id')[col].transform('mean').astype('float32')
    combined[f'l_{col}_std'] = combined.groupby('layout_id')[col].transform('std').astype('float32')
    combined[f'l_{col}_max'] = combined.groupby('layout_id')[col].transform('max').astype('float32')
    combined[f'l_{col}_p90'] = combined.groupby('layout_id')[col].transform(lambda x: x.quantile(0.9)).astype('float32')
    combined[f'l_{col}_ratio'] = (combined[col] / (combined[f'l_{col}_mean'] + 1e-6)).astype('float32')

n_layout_feats = len([c for c in combined.columns if c.startswith('l_')])
print(f"  Layout-level features: {n_layout_feats}", flush=True)

# --- C) Shift_hour-level aggregation ---
print("\n=== C) Shift_hour-level aggregation ===", flush=True)

HOUR_AGG_COLS = [
    'congestion_score', 'robot_active', 'order_inflow_15m',
    'pack_utilization', 'charge_queue_length', 'urgent_order_ratio',
]
HOUR_AGG_COLS = [c for c in HOUR_AGG_COLS if c in combined.columns]

combined['shift_hour_filled'] = combined['shift_hour'].fillna(-1)
for col in HOUR_AGG_COLS:
    combined[f'h_{col}_mean'] = combined.groupby('shift_hour_filled')[col].transform('mean').astype('float32')
    combined[f'h_{col}_std'] = combined.groupby('shift_hour_filled')[col].transform('std').astype('float32')
    combined[f'h_{col}_deviation'] = (combined[col] - combined[f'h_{col}_mean']).astype('float32')
combined = combined.drop(columns=['shift_hour_filled'])

n_hour_feats = len([c for c in combined.columns if c.startswith('h_')])
print(f"  Hour-level features: {n_hour_feats}", flush=True)

# --- D) Cross aggregation (layout × shift_hour) ---
print("\n=== D) Cross aggregation (layout x shift_hour) ===", flush=True)

CROSS_AGG_COLS = ['congestion_score', 'robot_active', 'order_inflow_15m', 'pack_utilization', 'charge_queue_length']
CROSS_AGG_COLS = [c for c in CROSS_AGG_COLS if c in combined.columns]

combined['shift_hour_filled'] = combined['shift_hour'].fillna(-1)
combined['lh_key'] = combined['layout_id'].astype(str) + '_' + combined['shift_hour_filled'].astype(str)
for col in CROSS_AGG_COLS:
    combined[f'lh_{col}_mean'] = combined.groupby('lh_key')[col].transform('mean').astype('float32')
    combined[f'lh_{col}_std'] = combined.groupby('lh_key')[col].transform('std').astype('float32')
combined = combined.drop(columns=['shift_hour_filled', 'lh_key'])

n_cross_feats = len([c for c in combined.columns if c.startswith('lh_')])
print(f"  Cross-level features: {n_cross_feats}", flush=True)

# --- E) Pair ratios ---
print("\n=== E) Pair ratios ===", flush=True)

combined['r_pack_per_robot'] = (combined['pack_station_count'] / (combined['robot_active'] + 1.0)).astype('float32')
combined['r_charger_per_robot'] = (combined['charger_count'] / (combined['robot_active'] + 1.0)).astype('float32')
combined['r_pack_util_x_inflow'] = (combined['pack_utilization'] * combined['order_inflow_15m'] / 100.0).astype('float32')
combined['r_congestion_x_robot_ratio'] = (combined['congestion_score'] * combined['robot_active'] / (combined['robot_total'] + 1.0)).astype('float32')
combined['r_queue_per_charger'] = (combined['charge_queue_length'] / (combined['charger_count'] + 1.0)).astype('float32')
combined['r_active_per_sqm'] = (combined['robot_active'] / (combined.get('floor_area_sqm', pd.Series(1000, index=combined.index)) / 1000.0 + 1.0)).astype('float32')
print(f"  Ratio features: 6", flush=True)

# --- Global NaN/inf cleanup ---
new_feat_cols = [c for c in combined.columns if c.startswith(('s_', 'l_', 'h_', 'lh_', 'r_'))]
for col in new_feat_cols:
    combined[col] = np.nan_to_num(combined[col].astype('float32').values, nan=0.0, posinf=1e6, neginf=-1e6)

print(f"\n=== Total new features before selection: {len(new_feat_cols)} ===", flush=True)

# ============================================================
# Part 3: Feature Selection (LGB importance)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Part 3: Feature Selection ===", flush=True)
print("=" * 60, flush=True)

# float32 conversion
float64_cols = combined.select_dtypes(include='float64').columns
combined[float64_cols] = combined[float64_cols].astype(np.float32)

combined = combined.sort_values('_original_idx').reset_index(drop=True)

# Prepare data for selection
y_train = combined.loc[combined['_is_train'] == 1, 'avg_delay_minutes_next_30m'].values
layout_ids_train = combined.loc[combined['_is_train'] == 1, 'layout_id'].values

all_new_feats = new_feat_cols
temp_feature_cols = base_feature_cols + [c for c in all_new_feats if c in combined.columns]
print(f"Temp total features: {len(temp_feature_cols)}", flush=True)

X_train_temp = combined.loc[combined['_is_train'] == 1, temp_feature_cols].fillna(0).astype('float32')

# Quick LGB for importance (1 fold for speed)
print("  Training selector LGB...", flush=True)
gkf = GroupKFold(n_splits=5)
tr_idx, va_idx = next(gkf.split(X_train_temp, y_train, groups=layout_ids_train))

lgb_selector = lgb.LGBMRegressor(
    objective='regression_l1', n_estimators=500, learning_rate=0.05,
    num_leaves=127, max_depth=8, min_child_samples=50,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
    verbosity=-1, random_state=42,
)
lgb_selector.fit(
    X_train_temp.iloc[tr_idx], y_train[tr_idx],
    eval_set=[(X_train_temp.iloc[va_idx], y_train[va_idx])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)

imp_df = pd.DataFrame({
    'feature': temp_feature_cols,
    'importance': lgb_selector.feature_importances_,
}).sort_values('importance', ascending=False).reset_index(drop=True)

new_imp = imp_df[imp_df['feature'].isin(all_new_feats)].copy()
new_imp['rank'] = range(1, len(new_imp) + 1)

print(f"\n=== Top 30 new features by importance ===", flush=True)
print(new_imp.head(30).to_string(index=False), flush=True)

# Select top 200
TOP_N_NEW = 200
selected_new_feats = new_imp.head(TOP_N_NEW)['feature'].tolist()
dropped_new_feats = [f for f in all_new_feats if f not in selected_new_feats and f in combined.columns]

print(f"\n=== Selection result ===", flush=True)
print(f"  Selected new features: {len(selected_new_feats)}", flush=True)
print(f"  Dropped new features: {len(dropped_new_feats)}", flush=True)

final_feature_cols = base_feature_cols + selected_new_feats
print(f"  Final total features: {len(final_feature_cols)}", flush=True)

# Drop unused features to save memory
combined = combined.drop(columns=[c for c in dropped_new_feats if c in combined.columns])
new_imp.to_csv('output/phase15_feature_importance.csv', index=False)

del X_train_temp, lgb_selector
gc.collect()

# ============================================================
# Part 4: Final Training (Phase 13s1 structure)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Part 4: Final Training ===", flush=True)
print("=" * 60, flush=True)

train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()
del combined
gc.collect()

feature_cols = final_feature_cols
X = train_fe[feature_cols]
y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y)
y_sqrt = np.sqrt(y)
groups = train_fe['layout_id']
time_idx = train_fe['implicit_timeslot'].values.astype(np.float32)
X_test = test_fe[feature_cols]

assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID order mismatch!"
print(f"Total features: {len(feature_cols)}", flush=True)
print("ID order verified!", flush=True)

# CV
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(sgkf.split(X, y_binned, groups=groups))

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    tr_layouts = set(train_fe.iloc[tr_idx]['layout_id'].unique())
    va_layouts = set(train_fe.iloc[va_idx]['layout_id'].unique())
    assert len(tr_layouts & va_layouts) == 0
print("  CV validation passed!", flush=True)

# Sample weight (base only)
def build_sample_weight(y_arr, time_arr):
    w = np.ones(len(y_arr), dtype=np.float32)
    q90, q95, q99 = np.nanquantile(y_arr, [0.90, 0.95, 0.99])
    w += 0.15 * (y_arr >= q90).astype(np.float32)
    w += 0.30 * (y_arr >= q95).astype(np.float32)
    w += 0.60 * (y_arr >= q99).astype(np.float32)
    if time_arr is not None:
        w += 0.08 * (time_arr / 24.0).astype(np.float32)
    return w

sample_w = build_sample_weight(y.values, time_idx)

# NN Preprocessing
scaler = StandardScaler()
X_train_nn = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_test_nn = np.nan_to_num(X_test.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_train_nn = scaler.fit_transform(X_train_nn).astype(np.float32)
X_test_nn = scaler.transform(X_test_nn).astype(np.float32)
y_log_nn = y_log.values.astype(np.float32)

# --- Tree 6 Models ---
print("\n=== Tree 6 Models ===", flush=True)
oof_preds = {}
test_preds = {}
cv_maes = {}

# Model 1: LGB raw+MAE
print("\n  [Model 1] LGB raw+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase15_lgb_raw.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m1_oof, m1_test, cv_maes['lgb_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
    m1_models = []
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
ckpt_path = 'output/ckpt_phase15_lgb_huber.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m2_oof, m2_test, cv_maes['lgb_log1p_huber'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
    m2_oof = np.zeros(len(X), dtype=np.float32)
    m2_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(objective='huber', huber_delta=0.9, n_estimators=2000, learning_rate=0.03,
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
ckpt_path = 'output/ckpt_phase15_lgb_sqrt.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m3_oof, m3_test, cv_maes['lgb_sqrt_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
    m3_oof = np.zeros(len(X), dtype=np.float32)
    m3_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(objective='mae', n_estimators=2000, learning_rate=0.03,
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
ckpt_path = 'output/ckpt_phase15_xgb.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m4_oof, m4_test, cv_maes['xgb_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
    m4_oof = np.zeros(len(X), dtype=np.float32)
    m4_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8,
                              min_child_weight=6, subsample=0.9, colsample_bytree=0.85,
                              reg_lambda=1.5, reg_alpha=0.05, objective='reg:absoluteerror',
                              eval_metric='mae', tree_method='hist', random_state=42, verbosity=0,
                              early_stopping_rounds=100)
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

# Model 5: Cat log1p+MAE
print("\n  [Model 5] Cat log1p+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase15_cat_log1p.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m5_oof, m5_test, cv_maes['cat_log1p_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
    m5_oof = np.zeros(len(X), dtype=np.float32)
    m5_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        train_pool = Pool(X.iloc[tr_idx], y_log.iloc[tr_idx], weight=sample_w[tr_idx])
        eval_pool = Pool(X.iloc[va_idx], y_log.iloc[va_idx])
        m = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
                               subsample=0.9, loss_function='MAE', random_seed=42, verbose=0)
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

# Model 6: Cat raw+MAE
print("\n  [Model 6] Cat raw+MAE...", flush=True)
ckpt_path = 'output/ckpt_phase15_cat_raw.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    m6_oof, m6_test, cv_maes['cat_raw_mae'] = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
    m6_oof = np.zeros(len(X), dtype=np.float32)
    m6_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], weight=sample_w[tr_idx])
        eval_pool = Pool(X.iloc[va_idx], y.iloc[va_idx])
        m = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                               subsample=0.85, loss_function='MAE', random_seed=42, verbose=0)
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

# --- NN: Keras MLP ---
print("\n=== NN: Keras MLP ===", flush=True)
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
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mae', metrics=['mae'])
    return model

ckpt_path = 'output/ckpt_phase15_mlp.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    mlp_oof, mlp_test, mlp_cv = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
    mlp_oof = np.zeros(len(X), dtype=np.float32)
    mlp_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        tf.random.set_seed(42); np.random.seed(42)
        model = build_mlp(X_train_nn.shape[1])
        es = keras_callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        rlr = keras_callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
        model.fit(X_train_nn[tr_idx], y_log_nn[tr_idx],
                  validation_data=(X_train_nn[va_idx], y_log_nn[va_idx]),
                  epochs=100, batch_size=512, callbacks=[es, rlr], verbose=0)
        pred = np.clip(np.expm1(model.predict(X_train_nn[va_idx], verbose=0).flatten()), 0, None).astype(np.float32)
        mlp_oof[va_idx] = pred
        mlp_test += np.clip(np.expm1(model.predict(X_test_nn, verbose=0).flatten()), 0, None).astype(np.float32) / 5
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.values[va_idx], pred):.4f}", flush=True)
        del model; tf.keras.backend.clear_session(); gc.collect()
    mlp_cv = mean_absolute_error(y, mlp_oof)
    save_ckpt(ckpt_path, {'oof': mlp_oof, 'test': mlp_test, 'cv_mae': mlp_cv})
oof_preds['keras_mlp'] = mlp_oof
test_preds['keras_mlp'] = mlp_test
cv_maes['keras_mlp'] = mlp_cv
print(f"  Keras MLP CV MAE: {mlp_cv:.4f}", flush=True)

# --- NN: TabNet ---
print("\n=== NN: TabNet ===", flush=True)
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

ckpt_path = 'output/ckpt_phase15_tabnet.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    tabnet_oof, tabnet_test, tabnet_cv = ckpt['oof'], ckpt['test'], ckpt['cv_mae']
else:
    tabnet_oof = np.zeros(len(X), dtype=np.float32)
    tabnet_test = np.zeros(len(X_test), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        torch.manual_seed(42); np.random.seed(42)
        model = TabNetRegressor(n_d=32, n_a=32, n_steps=5, gamma=1.5, n_independent=2, n_shared=2,
                                 lambda_sparse=1e-4, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                                 scheduler_params={"step_size": 10, "gamma": 0.9},
                                 scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax', seed=42, verbose=10)
        model.fit(X_train_nn[tr_idx], y_log_nn[tr_idx].reshape(-1, 1),
                  eval_set=[(X_train_nn[va_idx], y_log_nn[va_idx].reshape(-1, 1))],
                  eval_metric=['mae'], max_epochs=100, patience=15, batch_size=2048, virtual_batch_size=256)
        pred = np.clip(np.expm1(model.predict(X_train_nn[va_idx]).flatten()), 0, None).astype(np.float32)
        tabnet_oof[va_idx] = pred
        tabnet_test += np.clip(np.expm1(model.predict(X_test_nn).flatten()), 0, None).astype(np.float32) / 5
        print(f"    Fold {fold_i+1} MAE: {mean_absolute_error(y.values[va_idx], pred):.4f}", flush=True)
        del model; gc.collect()
    tabnet_cv = mean_absolute_error(y, tabnet_oof)
    save_ckpt(ckpt_path, {'oof': tabnet_oof, 'test': tabnet_test, 'cv_mae': tabnet_cv})
oof_preds['tabnet'] = tabnet_oof
test_preds['tabnet'] = tabnet_test
cv_maes['tabnet'] = tabnet_cv
print(f"  TabNet CV MAE: {tabnet_cv:.4f}", flush=True)

# --- Level 1 + Level 2 ---
print("\n=== Level 1 Ensemble ===", flush=True)
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
    res = minimize(obj, x0=np.ones(n)/n, method='Nelder-Mead', options={'maxiter': 10000})
    w = res.x / res.x.sum()
    return w, mean_absolute_error(y_true, oof_mat @ w)

tree_w, tree_cv = opt_weights(tree_oof_matrix, y)
print(f"  Tree only Level 1 CV MAE: {tree_cv:.4f}", flush=True)
all_w, all_cv = opt_weights(oof_matrix, y)
print(f"  Tree+NN Level 1 CV MAE: {all_cv:.4f}", flush=True)

print("\n=== Level 2 LGB Stacking ===", flush=True)
meta_cols_extra = ['implicit_timeslot', 'order_inflow_15m', 'battery_mean',
                   'robot_active', 'pack_utilization', 'congestion_score']
meta_train_extra = train_fe[meta_cols_extra].values.astype(np.float32)
meta_test_extra = test_fe[meta_cols_extra].values.astype(np.float32)
meta_X_train = np.nan_to_num(np.column_stack([oof_matrix, meta_train_extra]), nan=0.0)
meta_X_test = np.nan_to_num(np.column_stack([test_matrix, meta_test_extra]), nan=0.0)
meta_feature_names = model_names + meta_cols_extra
meta_X_train_df = pd.DataFrame(meta_X_train, columns=meta_feature_names)
meta_X_test_df = pd.DataFrame(meta_X_test, columns=meta_feature_names)

meta_lgb_oof = np.zeros(len(X), dtype=np.float32)
meta_lgb_test = np.zeros(len(meta_X_test), dtype=np.float32)
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=16, max_depth=4,
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

# Final selection
candidates = {
    'tree_only_l1': (tree_cv, tree_oof_matrix @ tree_w, tree_test_matrix @ tree_w),
    'all_l1': (all_cv, oof_matrix @ all_w, test_matrix @ all_w),
    'lgb_stack': (meta_lgb_cv, meta_lgb_oof, meta_lgb_test),
}
best_name = min(candidates, key=lambda k: candidates[k][0])
final_cv = candidates[best_name][0]
final_oof = candidates[best_name][1]
final_test = candidates[best_name][2]
print(f"\n  Final: {best_name} (CV MAE: {final_cv:.4f})", flush=True)

# ============================================================
# Part 5: Results
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 15 Results ===", flush=True)
print("=" * 60, flush=True)
print(f"Base features (Phase 13s1): {len(base_feature_cols)}", flush=True)
print(f"New aggregation features: {len(selected_new_feats)}", flush=True)
print(f"Total: {len(final_feature_cols)}", flush=True)
print("Tree models:", flush=True)
for n in tree_names:
    print(f"  {n:20s}: {cv_maes[n]:.4f}", flush=True)
print(f"  {'keras_mlp':20s}: {cv_maes['keras_mlp']:.4f}", flush=True)
print(f"  {'tabnet':20s}: {cv_maes['tabnet']:.4f}", flush=True)
print(f"Level 1 tree:     {tree_cv:.4f}", flush=True)
print(f"Level 1 all:      {all_cv:.4f}", flush=True)
print(f"Level 2 stack:    {meta_lgb_cv:.4f}", flush=True)
print(f"Final:            {best_name} ({final_cv:.4f})", flush=True)
print(f"\n=== Comparison ===", flush=True)
print(f"Phase 12a CV:     8.5754", flush=True)
print(f"Phase 13s1 CV:    8.5668  Public 10.0078", flush=True)
print(f"Phase 15 CV:      {final_cv:.4f}", flush=True)
print(f"Improvement:      {8.5668 - final_cv:+.4f}", flush=True)

# Hard layout analysis
if os.path.exists('output/phase13s2_analysis/layout_mae_ranking.csv'):
    hard_df = pd.read_csv('output/phase13s2_analysis/layout_mae_ranking.csv')
    if 'difficulty' in hard_df.columns:
        hard_ids = hard_df[hard_df['difficulty'] == 'hard']['layout_id'].tolist()
        hard_mask = train_fe['layout_id'].isin(hard_ids).values
        hard_mae = np.abs(final_oof[hard_mask] - y.values[hard_mask]).mean()
        print(f"\nHard layout MAE: {hard_mae:.4f} (baseline 18.784)", flush=True)

# ============================================================
# Part 6: Submission
# ============================================================
print("\n=== Submission ===", flush=True)
final_test = np.clip(final_test, 0, None)
submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase15.csv', index=False)
drive_sub_path = os.path.join(DRIVE_CKPT_DIR, 'submission_phase15.csv')
submission.to_csv(drive_sub_path, index=False)

assert list(submission.columns) == list(sample_sub.columns)
assert len(submission) == len(sample_sub)
assert (submission['ID'] == sample_sub['ID']).all()
assert (submission['avg_delay_minutes_next_30m'] >= 0).all()
print("submission_phase15.csv done", flush=True)
print(submission.describe(), flush=True)

# Feature importance
if len(m1_models) > 0:
    importance = pd.DataFrame({'feature': feature_cols, 'importance': m1_models[0].feature_importances_}).sort_values('importance', ascending=False)
    top30 = importance.head(30).sort_values('importance')
    n_new_in_top30 = top30['feature'].isin(selected_new_feats).sum()
    print(f"\n  New features in Top 30: {n_new_in_top30}", flush=True)

    colors = ['crimson' if f in selected_new_feats else ('orangered' if f.startswith('q_') else 'steelblue') for f in top30['feature']]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top30['feature'], top30['importance'], color=colors)
    ax.set_title('Feature Importance Top 30 (Phase 15)')
    legend_elements = [Patch(facecolor='steelblue', label='Base'), Patch(facecolor='orangered', label='Queueing'),
                       Patch(facecolor='crimson', label='New Agg')]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig('output/feature_importance_phase15.png', dpi=150, bbox_inches='tight')
    print("  feature_importance_phase15.png saved", flush=True)

print("\n=== Phase 15 Complete ===", flush=True)
