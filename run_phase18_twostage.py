import pandas as pd
import numpy as np
import gc
import pickle
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

DRIVE_CKPT_DIR = '/content/drive/MyDrive/dacon_ckpt'
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

PIPELINE_VERSION = "phase18_v1"


def save_ckpt(local_path, data, feature_cols=None):
    if feature_cols is not None:
        data['feature_cols'] = list(feature_cols)
        data['n_features'] = len(feature_cols)
    data['pipeline_version'] = PIPELINE_VERSION
    with open(local_path, 'wb') as f:
        pickle.dump(data, f)
    drive_path = os.path.join(DRIVE_CKPT_DIR, os.path.basename(local_path))
    if os.path.exists(os.path.dirname(drive_path)):
        shutil.copy(local_path, drive_path)
    print(f"  Saved: {local_path}", flush=True)


def load_ckpt(local_path, expected_features=None):
    drive_path = os.path.join(DRIVE_CKPT_DIR, os.path.basename(local_path))
    for path in [drive_path, local_path]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                ckpt = pickle.load(f)
            if ckpt.get('pipeline_version') != PIPELINE_VERSION:
                print(f"  {os.path.basename(path)}: version mismatch, invalidating cache", flush=True)
                return None
            if expected_features is not None:
                cached_fc = ckpt.get('feature_cols')
                if cached_fc is not None and cached_fc != list(expected_features):
                    print(f"  {os.path.basename(path)}: feature_cols mismatch, invalidating cache", flush=True)
                    return None
            return ckpt
    return None


# ##############################################################
# Part 1A: Phase 13s1 base features (346)
# ##############################################################
print("=" * 60, flush=True)
print("=== Part 1A: Phase 13s1 Base Features ===", flush=True)
print("=" * 60, flush=True)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
N_TRAIN = len(train)
train['_is_train'] = 1; test['_is_train'] = 0
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined['_original_idx'] = range(len(combined))
del train, test; gc.collect()

combined['implicit_timeslot'] = combined.groupby('scenario_id').cumcount()
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)

# Time Series (64)
print("=== Time Series Features ===", flush=True)
ts_cols = ['battery_mean', 'low_battery_ratio', 'order_inflow_15m', 'congestion_score',
           'robot_idle', 'robot_charging', 'max_zone_density', 'pack_utilization']
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
print("  64 done", flush=True)

# Layout join + basic
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

# Phase 3B Interactions (8)
combined['orders_per_packstation'] = combined['order_inflow_15m'] / combined['pack_station_count'].replace(0, np.nan)
combined['pack_dock_pressure'] = combined['pack_utilization'] * combined['loading_dock_util']
combined['dock_wait_pressure'] = combined['outbound_truck_wait_min'] * combined['loading_dock_util']
combined['shift_load_pressure'] = combined['prev_shift_volume'] * combined['order_inflow_15m']
combined['battery_congestion'] = combined['low_battery_ratio'] * combined['congestion_score']
combined['storage_density_congestion'] = combined['storage_density_pct'] * combined['congestion_score']
combined['battery_trip_pressure'] = combined['low_battery_ratio'] * combined['avg_trip_distance']
combined['demand_density'] = combined['order_inflow_15m'] * combined['max_zone_density']

# Missing indicators
train_part = combined[combined['_is_train'] == 1]
missing_counts = train_part.isnull().sum().sort_values(ascending=False)
top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith('_')]
for col in top10_missing:
    if col in combined.columns:
        combined[f'{col}_missing'] = combined[col].isnull().astype(int)
del train_part; gc.collect()

# Onset (8)
print("=== Onset ===", flush=True)
def compute_onset_idx(group, flag_col, timeslot_col='implicit_timeslot'):
    flags, slots = group[flag_col].values, group[timeslot_col].values
    result = np.full(len(group), -1, dtype=np.float32)
    first_idx = -1
    for i in range(1, len(flags)):
        if flags[i-1] > 0 and first_idx == -1: first_idx = slots[i-1]
        result[i] = first_idx
    return pd.Series(result, index=group.index)

for flag_name, flag_col in [('charging', 'robot_charging'), ('queue', 'charge_queue_length'), ('congestion', 'congestion_score')]:
    combined[f'_{flag_name}_flag'] = (combined[flag_col] > 0).astype(np.float32)
    combined[f'_{flag_name}_flag_prev'] = combined.groupby('scenario_id')[f'_{flag_name}_flag'].shift(1).fillna(0)
    combined[f'{flag_name}_ever_started'] = combined.groupby('scenario_id')[f'_{flag_name}_flag_prev'].cummax()
    combined[f'{flag_name}_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(lambda g: compute_onset_idx(g, f'_{flag_name}_flag'))
combined['charging_steps_since_start'] = np.where(combined['charging_start_idx'] >= 0, combined['implicit_timeslot'] - combined['charging_start_idx'], -1).astype(np.float32)
combined['charging_started_early'] = ((combined['charging_start_idx'] >= 0) & (combined['charging_start_idx'] < 5)).astype(np.float32)
combined.drop(columns=[c for c in combined.columns if c.startswith('_') and ('_flag' in c)], inplace=True)
print("  8 done", flush=True)

# Expanding Mean (30)
print("=== Expanding Mean ===", flush=True)
expanding_cols_mean = ['order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio', 'heavy_item_ratio', 'robot_active', 'battery_mean', 'low_battery_ratio', 'congestion_score', 'max_zone_density', 'pack_utilization', 'loading_dock_util', 'charge_queue_length', 'fault_count_15m', 'avg_trip_distance']
for i, col in enumerate(expanding_cols_mean):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    expmean = shifted.groupby(combined['scenario_id']).expanding().mean().droplevel(0).sort_index()
    combined[f'{col}_expmean_prev'] = expmean.astype(np.float32)
    combined[f'{col}_delta_expmean'] = (combined[col] - combined[f'{col}_expmean_prev']).astype(np.float32)
    if (i + 1) % 5 == 0: print(f"  {i+1}/{len(expanding_cols_mean)}", flush=True)

# Nonlinear (7)
combined['battery_mean_below_44'] = np.maximum(44.0 - combined['battery_mean'], 0).astype(np.float32)
combined['low_battery_ratio_above_02'] = np.maximum(combined['low_battery_ratio'] - 0.2, 0).astype(np.float32)
combined['pack_utilization_sq'] = (combined['pack_utilization'] ** 2).astype(np.float32)
combined['loading_dock_util_sq'] = (combined['loading_dock_util'] ** 2).astype(np.float32)
combined['congestion_score_sq'] = (combined['congestion_score'] ** 2).astype(np.float32)
combined['charge_pressure_nl'] = ((combined['robot_charging'] + combined['charge_queue_length']) / (combined['charger_count'] + 1)).astype(np.float32)
combined['charge_pressure_nl_sq'] = (combined['charge_pressure_nl'] ** 2).astype(np.float32)

# Phase (6)
combined['is_early_phase'] = (combined['implicit_timeslot'] <= 5).astype(np.float32)
combined['is_mid_phase'] = ((combined['implicit_timeslot'] >= 6) & (combined['implicit_timeslot'] <= 15)).astype(np.float32)
combined['is_late_phase'] = (combined['implicit_timeslot'] >= 16).astype(np.float32)
combined['time_frac'] = (combined['implicit_timeslot'] / 24.0).astype(np.float32)
combined['time_remaining'] = (24 - combined['implicit_timeslot']).astype(np.float32)
combined['time_frac_sq'] = (combined['time_frac'] ** 2).astype(np.float32)

# Competitor (54)
print("=== Competitor ===", flush=True)
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
numeric_cols = [c for c in combined.select_dtypes(include=[np.number]).columns if not c.startswith('_')]
combined['n_missing_all'] = combined[numeric_cols].isnull().sum(axis=1).astype(np.float32)
dyn_cols_exist = [c for c in ['order_inflow_15m', 'battery_mean', 'battery_std', 'low_battery_ratio', 'robot_active', 'robot_idle', 'robot_charging', 'congestion_score', 'max_zone_density', 'pack_utilization', 'loading_dock_util', 'charge_queue_length', 'fault_count_15m', 'avg_trip_distance', 'unique_sku_15m'] if c in combined.columns]
combined['n_missing_dynamic'] = combined[dyn_cols_exist].isnull().sum(axis=1).astype(np.float32)
combined['missing_ratio'] = (combined['n_missing_all'] / len(numeric_cols)).astype(np.float32)

print("  rolling max + deviation...", flush=True)
for col in ['order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization', 'loading_dock_util']:
    shifted = combined.groupby('scenario_id')[col].shift(1)
    rollmax3 = shifted.groupby(combined['scenario_id']).rolling(3, min_periods=1).max().droplevel(0).sort_index()
    combined[f'{col}_rollmax3_prev'] = rollmax3.astype(np.float32)
    combined[f'{col}_dev_rollmax3'] = (combined[col] - combined[f'{col}_rollmax3_prev']).astype(np.float32)

# Expanding Extension (20)
print("=== Expanding Extension ===", flush=True)
for i, col in enumerate(['order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization', 'loading_dock_util', 'robot_active', 'low_battery_ratio', 'avg_trip_distance', 'unique_sku_15m', 'max_zone_density']):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    grp = shifted.groupby(combined['scenario_id'])
    combined[f'{col}_expstd_prev'] = grp.expanding().std().droplevel(0).sort_index().astype(np.float32)
    combined[f'{col}_expmax_prev'] = grp.expanding().max().droplevel(0).sort_index().astype(np.float32)
    if (i + 1) % 5 == 0: print(f"  {i+1}/10", flush=True)

# Queueing Theory (42)
print("=== Queueing Theory ===", flush=True)
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
ar = (combined['order_inflow_15m'] / 15.0).astype(np.float32)
combined['q_arrival_rate'] = ar
combined['q_expected_charge_wait'] = (combined['charge_queue_length'] / (ar + EPS)).astype(np.float32)
combined['q_effective_service_robot'] = (combined['robot_active'] / (combined['avg_trip_distance'] + EPS)).astype(np.float32)
combined['q_arrival_service_gap'] = (ar - combined['q_effective_service_robot']).astype(np.float32)
combined['q_throughput_robot'] = (combined['robot_active'] * (1.0 - combined['congestion_score'])).astype(np.float32)
combined['q_throughput_pack'] = (combined['pack_station_count'] * combined['pack_utilization']).astype(np.float32)
stg = combined[['q_rho_robot', 'q_rho_charger', 'q_rho_pack', 'q_rho_loading']]
combined['q_bottleneck_max'] = stg.max(axis=1).astype(np.float32)
combined['q_bottleneck_min'] = stg.min(axis=1).astype(np.float32)
combined['q_bottleneck_mean'] = stg.mean(axis=1).astype(np.float32)
combined['q_bottleneck_std'] = stg.std(axis=1).astype(np.float32)
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
for col in [c for c in combined.columns if c.startswith('q_')]:
    combined[col] = np.nan_to_num(combined[col].values, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

# EDA (7)
combined['shift_hour'] = combined['shift_hour'].fillna(-1)
combined['is_night_shift'] = (combined['shift_hour'] >= 18).astype('float32')
combined['hours_after_peak'] = (combined['shift_hour'] - 17).clip(lower=0).astype('float32')
combined['shift_phase'] = pd.cut(combined['shift_hour'], bins=[-2, 6, 12, 17, 23], labels=[0, 1, 2, 3]).astype('float32').fillna(0).astype('float32')
combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)
combined['time_step_in_scenario'] = combined.groupby('scenario_id').cumcount().astype('float32')
combined['ts_squared'] = (combined['time_step_in_scenario'] ** 2).astype('float32')
combined['inflow_per_active_robot'] = (combined['order_inflow_15m'] / (combined['robot_active'] + 1.0)).astype('float32')
combined['is_overload_not_congestion'] = ((combined['order_inflow_15m'] > 130) & (combined['congestion_score'] < 8)).astype('float32')
for feat in ['is_night_shift', 'hours_after_peak', 'shift_phase', 'time_step_in_scenario', 'ts_squared', 'inflow_per_active_robot', 'is_overload_not_congestion']:
    combined[feat] = np.nan_to_num(combined[feat].values, nan=0.0).astype(np.float32)

# Drop low-corr layout
for c in ['layout_compactness', 'zone_dispersion', 'one_way_ratio', 'aisle_width_avg', 'floor_area_sqm', 'ceiling_height_m', 'building_age_years', 'intersection_count', 'fire_sprinkler_count', 'emergency_exit_count', 'robot_per_area', 'congestion_per_width', 'zone_density_per_width', 'order_per_sqm', 'warehouse_volume', 'intersection_density', 'pack_station_density', 'charger_density', 'robot_density_layout', 'movement_friction', 'layout_compact_x_dispersion', 'one_way_friction']:
    if c in combined.columns: combined.drop(columns=[c], inplace=True)

layout_cols_surviving = [c for c in combined.columns if c in [
    'robot_total', 'pack_station_count', 'charger_count',
    'floor_area_sqm', 'ceiling_height_m', 'building_age_years',
    'aisle_width_avg', 'intersection_count', 'one_way_ratio',
    'layout_compactness', 'zone_dispersion', 'sku_concentration',
    'emergency_exit_count', 'fire_sprinkler_count',
]]
print(f"\nSurviving layout columns: {layout_cols_surviving}", flush=True)

drop_cols_meta = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']
base_feature_cols = [c for c in combined.columns if c not in drop_cols_meta and combined[c].dtype in [np.float32, np.float64, np.int32, np.int64, np.int8]]
print(f"  Base features: {len(base_feature_cols)}", flush=True)

# ##############################################################
# Part 1B: Phase 15 Aggregation Features (identical reproduction)
# ##############################################################
print("\n=== Part 1B: Phase 15 Aggregation ===", flush=True)

SCENARIO_AGG_COLS = [c for c in ['congestion_score', 'max_zone_density', 'robot_charging', 'low_battery_ratio', 'robot_idle', 'order_inflow_15m', 'battery_mean', 'charge_queue_length', 'avg_charge_wait', 'near_collision_15m', 'blocked_path_15m', 'unique_sku_15m', 'fault_count_15m', 'avg_recovery_time', 'sku_concentration', 'urgent_order_ratio', 'battery_std', 'robot_utilization', 'robot_active', 'pack_utilization', 'loading_dock_util', 'heavy_item_ratio', 'manual_override_ratio', 'cold_chain_ratio', 'outbound_truck_wait_min', 'packaging_material_cost'] if c in combined.columns]
for col in SCENARIO_AGG_COLS:
    for func in ['mean', 'std', 'max', 'min', 'median', 'skew']:
        if func == 'skew':
            combined[f's_{col}_{func}'] = combined.groupby('scenario_id')[col].transform(lambda x: x.skew() if len(x) > 2 else 0).astype('float32')
        else:
            combined[f's_{col}_{func}'] = combined.groupby('scenario_id')[col].transform(func).astype('float32')
for col in SCENARIO_AGG_COLS[:15]:
    combined[f's_{col}_first'] = combined.groupby('scenario_id')[col].transform('first').astype('float32')
    combined[f's_{col}_last'] = combined.groupby('scenario_id')[col].transform('last').astype('float32')
    combined[f's_{col}_trend'] = (combined[f's_{col}_last'] - combined[f's_{col}_first']).astype('float32')
    combined[f's_{col}_dist_from_max'] = (combined[f's_{col}_max'] - combined[col]).astype('float32')
    combined[f's_{col}_rank_in_scenario'] = combined.groupby('scenario_id')[col].rank(pct=True).astype('float32')
print(f"  Scenario agg: {len([c for c in combined.columns if c.startswith('s_')])}", flush=True)

LAYOUT_AGG_COLS = [c for c in ['congestion_score', 'robot_active', 'robot_charging', 'order_inflow_15m', 'pack_utilization', 'charge_queue_length', 'urgent_order_ratio', 'max_zone_density', 'battery_mean', 'robot_utilization', 'sku_concentration'] if c in combined.columns]
for col in LAYOUT_AGG_COLS:
    combined[f'l_{col}_mean'] = combined.groupby('layout_id')[col].transform('mean').astype('float32')
    combined[f'l_{col}_std'] = combined.groupby('layout_id')[col].transform('std').astype('float32')
    combined[f'l_{col}_max'] = combined.groupby('layout_id')[col].transform('max').astype('float32')
    combined[f'l_{col}_p90'] = combined.groupby('layout_id')[col].transform(lambda x: x.quantile(0.9)).astype('float32')
    combined[f'l_{col}_ratio'] = (combined[col] / (combined[f'l_{col}_mean'] + 1e-6)).astype('float32')

combined['shift_hour_filled'] = combined['shift_hour'].fillna(-1)
HOUR_AGG_COLS = [c for c in ['congestion_score', 'robot_active', 'order_inflow_15m', 'pack_utilization', 'charge_queue_length', 'urgent_order_ratio'] if c in combined.columns]
for col in HOUR_AGG_COLS:
    combined[f'h_{col}_mean'] = combined.groupby('shift_hour_filled')[col].transform('mean').astype('float32')
    combined[f'h_{col}_std'] = combined.groupby('shift_hour_filled')[col].transform('std').astype('float32')
    combined[f'h_{col}_deviation'] = (combined[col] - combined[f'h_{col}_mean']).astype('float32')

CROSS_AGG_COLS = [c for c in ['congestion_score', 'robot_active', 'order_inflow_15m', 'pack_utilization', 'charge_queue_length'] if c in combined.columns]
combined['lh_key'] = combined['layout_id'].astype(str) + '_' + combined['shift_hour_filled'].astype(str)
for col in CROSS_AGG_COLS:
    combined[f'lh_{col}_mean'] = combined.groupby('lh_key')[col].transform('mean').astype('float32')
    combined[f'lh_{col}_std'] = combined.groupby('lh_key')[col].transform('std').astype('float32')
combined.drop(columns=['shift_hour_filled', 'lh_key'], inplace=True)

combined['r_pack_per_robot'] = (combined['pack_station_count'] / (combined['robot_active'] + 1.0)).astype('float32')
combined['r_charger_per_robot'] = (combined['charger_count'] / (combined['robot_active'] + 1.0)).astype('float32')
combined['r_pack_util_x_inflow'] = (combined['pack_utilization'] * combined['order_inflow_15m'] / 100.0).astype('float32')
combined['r_congestion_x_robot_ratio'] = (combined['congestion_score'] * combined['robot_active'] / (combined['robot_total'] + 1.0)).astype('float32')
combined['r_queue_per_charger'] = (combined['charge_queue_length'] / (combined['charger_count'] + 1.0)).astype('float32')
combined['r_active_per_sqm'] = (combined['robot_active'] / 2.0).astype('float32')

p15_agg_feats = [c for c in combined.columns if c.startswith(('s_', 'l_', 'h_', 'lh_', 'r_'))]
for col in p15_agg_feats:
    combined[col] = np.nan_to_num(combined[col].astype('float32').values, nan=0.0, posinf=1e6, neginf=-1e6)
print(f"  Phase 15 agg total: {len(p15_agg_feats)}", flush=True)

# Phase 15 Feature Selection (identical reproduction)
print("\n=== Phase 15 Feature Selection ===", flush=True)
float64_cols = combined.select_dtypes(include='float64').columns
combined[float64_cols] = combined[float64_cols].astype(np.float32)
combined = combined.sort_values('_original_idx').reset_index(drop=True)

train_df = combined[combined['_is_train'] == 1].copy()
test_df = combined[combined['_is_train'] == 0].copy()
y_train = train_df['avg_delay_minutes_next_30m'].values
layout_ids_train = train_df['layout_id'].values
temp_fc = base_feature_cols + [c for c in p15_agg_feats if c in combined.columns]
X_tmp = combined.loc[combined['_is_train'] == 1, temp_fc].fillna(0).astype('float32')

gkf = GroupKFold(n_splits=5)
tr_i, va_i = next(gkf.split(X_tmp, y_train, groups=layout_ids_train))
sel = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, learning_rate=0.05, num_leaves=127, max_depth=8, min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, verbosity=-1, random_state=42, deterministic=True, force_col_wise=True, n_jobs=1)
sel.fit(X_tmp.iloc[tr_i], y_train[tr_i], eval_set=[(X_tmp.iloc[va_i], y_train[va_i])], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
imp_p15 = pd.DataFrame({'feature': temp_fc, 'importance': sel.feature_importances_}).sort_values('importance', ascending=False)
new_imp_p15 = imp_p15[imp_p15['feature'].isin(p15_agg_feats)]
p15_selected = new_imp_p15.head(200)['feature'].tolist()
p15_dropped = [f for f in p15_agg_feats if f not in p15_selected and f in combined.columns]
combined.drop(columns=p15_dropped, inplace=True)
feature_cols_p15 = base_feature_cols + p15_selected
print(f"  Phase 15 final: {len(feature_cols_p15)} features", flush=True)
del X_tmp, sel; gc.collect()

# ##############################################################
# Part 1C: Phase 16 2nd-order Features (identical reproduction)
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 1C: Phase 16 2nd-order Features ===", flush=True)
print("=" * 60, flush=True)

combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)

# A) Lag Features
print("\n=== A) Lag Features ===", flush=True)
LAG_COLS = [c for c in ['congestion_score', 'robot_active', 'pack_utilization', 'order_inflow_15m', 'charge_queue_length', 'battery_mean', 'robot_charging', 'max_zone_density', 'unique_sku_15m', 'outbound_truck_wait_min', 'low_battery_ratio', 'avg_charge_wait', 'urgent_order_ratio', 'sku_concentration'] if c in combined.columns]
for col in LAG_COLS:
    for lag in [1, 2, 3]:
        combined[f'lag{lag}_{col}'] = combined.groupby('scenario_id')[col].shift(lag).astype('float32')
print(f"  Lag features: {len(LAG_COLS) * 3}", flush=True)

# B) Rolling Features
print("=== B) Rolling Features ===", flush=True)
ROLL_COLS = [c for c in ['congestion_score', 'robot_active', 'pack_utilization', 'order_inflow_15m', 'charge_queue_length', 'battery_mean', 'robot_charging', 'max_zone_density', 'urgent_order_ratio', 'unique_sku_15m'] if c in combined.columns]
for col in ROLL_COLS:
    for window in [3, 5]:
        combined[f'roll{window}_{col}'] = combined.groupby('scenario_id')[col].transform(lambda x: x.rolling(window, min_periods=1).mean()).astype('float32')
        combined[f'roll{window}std_{col}'] = combined.groupby('scenario_id')[col].transform(lambda x: x.rolling(window, min_periods=1).std()).astype('float32')
print(f"  Rolling features: {len(ROLL_COLS) * 4}", flush=True)

# C) Cumulative Features
print("=== C) Cumulative Features ===", flush=True)
CUM_COLS = [c for c in ['congestion_score', 'robot_active', 'pack_utilization', 'order_inflow_15m', 'charge_queue_length', 'max_zone_density', 'urgent_order_ratio'] if c in combined.columns]
for col in CUM_COLS:
    combined[f'cummax_{col}'] = combined.groupby('scenario_id')[col].cummax().astype('float32')
    combined[f'cummin_{col}'] = combined.groupby('scenario_id')[col].cummin().astype('float32')
    combined[f'cummean_{col}'] = combined.groupby('scenario_id')[col].transform(lambda x: x.expanding().mean()).astype('float32')
    combined[f'gap_from_cummax_{col}'] = (combined[f'cummax_{col}'] - combined[col]).astype('float32')
print(f"  Cumulative features: {len(CUM_COLS) * 4}", flush=True)

# D) Interaction Features
print("=== D) Interaction Features ===", flush=True)
combined['ix_pack_x_inflow'] = (combined['pack_utilization'] * combined['order_inflow_15m'] / 100.0).astype('float32')
combined['ix_pack_x_congestion'] = (combined['pack_utilization'] * combined['congestion_score']).astype('float32')
combined['ix_pack_x_robot_active'] = (combined['pack_utilization'] * combined['robot_active'] / 30.0).astype('float32')
combined['ix_queue_x_low_bat'] = (combined['charge_queue_length'] * combined['low_battery_ratio']).astype('float32')
combined['ix_charge_x_charging'] = (combined['avg_charge_wait'] * combined['robot_charging']).astype('float32')
combined['ix_queue_per_charger'] = (combined['charge_queue_length'] / (combined['charger_count'] + 1.0)).astype('float32')
combined['ix_congestion_x_density'] = (combined['congestion_score'] * combined['max_zone_density']).astype('float32')
combined['ix_blocked_x_collision'] = (combined['blocked_path_15m'] * combined['near_collision_15m']).astype('float32')
combined['ix_urgent_x_heavy'] = (combined['urgent_order_ratio'] * combined['heavy_item_ratio']).astype('float32')
combined['ix_sku_x_inflow'] = (combined['unique_sku_15m'] * combined['order_inflow_15m'] / 1000.0).astype('float32')
combined['ix_cold_x_urgent'] = (combined['cold_chain_ratio'] * combined['urgent_order_ratio']).astype('float32')
combined['ix_pack_minus_truck_norm'] = (combined['pack_utilization'] - combined['outbound_truck_wait_min'] / 30.0).astype('float32')
combined['ix_active_minus_charging'] = (combined['robot_active'] - combined['robot_charging']).astype('float32')
combined['ix_charging_ratio'] = (combined['robot_charging'] / (combined['robot_active'] + combined['robot_charging'] + 1.0)).astype('float32')
combined['ix_active_per_pack'] = (combined['robot_active'] / (combined['pack_station_count'] + 1.0)).astype('float32')
print(f"  Interaction features: 15", flush=True)

# E) Lag-based Scenario Aggregation
print("=== E) Lag-based Scenario Agg ===", flush=True)
LAG_AGG_TARGETS = [c for c in ['lag1_battery_mean', 'lag1_charge_queue_length', 'lag1_robot_charging', 'lag1_congestion_score', 'lag1_order_inflow_15m', 'lag1_pack_utilization', 'lag1_max_zone_density'] if c in combined.columns]
for col in LAG_AGG_TARGETS:
    combined[f's_{col}_mean'] = combined.groupby('scenario_id')[col].transform('mean').astype('float32')
    combined[f's_{col}_std'] = combined.groupby('scenario_id')[col].transform('std').astype('float32')
    combined[f's_{col}_max'] = combined.groupby('scenario_id')[col].transform('max').astype('float32')
print(f"  Lag-agg features: {len(LAG_AGG_TARGETS) * 3}", flush=True)

# NaN cleanup
new_2nd_feats = [c for c in combined.columns if c.startswith(('lag', 'roll', 'cummax', 'cummin', 'cummean', 'gap_from', 'ix_', 's_lag'))]
for col in new_2nd_feats:
    combined[col] = np.nan_to_num(combined[col].astype('float32').values, nan=0.0, posinf=1e6, neginf=-1e6)
print(f"\n  Total new 2nd-order features: {len(new_2nd_feats)}", flush=True)

# Phase 16 2nd-order Feature Selection (identical reproduction)
print("\n=== Phase 16 2nd-order Feature Selection ===", flush=True)
combined = combined.sort_values('_original_idx').reset_index(drop=True)

temp_fc2 = feature_cols_p15 + [c for c in new_2nd_feats if c in combined.columns]
X_tmp2 = combined.loc[combined['_is_train'] == 1, temp_fc2].fillna(0).astype('float32')
print(f"  Temp total: {len(temp_fc2)}", flush=True)

tr_i2, va_i2 = next(gkf.split(X_tmp2, y_train, groups=layout_ids_train))
sel2 = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, learning_rate=0.05, num_leaves=127, max_depth=8, min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, verbosity=-1, random_state=42, deterministic=True, force_col_wise=True, n_jobs=1)
sel2.fit(X_tmp2.iloc[tr_i2], y_train[tr_i2], eval_set=[(X_tmp2.iloc[va_i2], y_train[va_i2])], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

imp2 = pd.DataFrame({'feature': temp_fc2, 'importance': sel2.feature_importances_}).sort_values('importance', ascending=False)
new_imp2 = imp2[imp2['feature'].isin(new_2nd_feats)].copy()
new_imp2['rank'] = range(1, len(new_imp2) + 1)
print(f"\n  Top 30 new 2nd-order:", flush=True)
print(new_imp2.head(30).to_string(index=False), flush=True)

TOP_N = 150
selected_2nd = new_imp2.head(TOP_N)['feature'].tolist()
dropped_2nd = [f for f in new_2nd_feats if f not in selected_2nd and f in combined.columns]
print(f"\n  Category breakdown (top {TOP_N}):", flush=True)
for prefix in ['lag', 'roll', 'cummax', 'cummin', 'cummean', 'gap_from', 'ix_', 's_lag']:
    cnt = sum(1 for f in selected_2nd if f.startswith(prefix))
    if cnt > 0: print(f"    {prefix}*: {cnt}", flush=True)

combined.drop(columns=dropped_2nd, inplace=True)
phase16_feature_cols = feature_cols_p15 + selected_2nd
print(f"\n  Phase 16 total features: {len(phase16_feature_cols)}", flush=True)
del X_tmp2, sel2; gc.collect()


# ##############################################################
# Part 2: Phase 17 NEW Features (5 categories, ~40)
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 2: Phase 17 NEW Features ===", flush=True)
print("=" * 60, flush=True)

combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)

# ---- Category A: Explicit Bottleneck Explosion (12) ----
print("\n=== A) Bottleneck Explosion Features ===", flush=True)

eps = 1e-3

# Pack station utilization
rho_pack = combined['pack_utilization'].fillna(0).clip(0, 1 - eps).astype('float32')

# Robot utilization (active / total)
rho_robot = (
    combined['robot_active'] / (combined['robot_total'] + 1)
).fillna(0).clip(0, 1 - eps).astype('float32')

# Charge bottleneck proxy
rho_charge = (
    combined['charge_queue_length'] / (combined['charger_count'] * 10 + 1)
).fillna(0).clip(0, 1 - eps).astype('float32')

# Truck bottleneck proxy (30min basis)
rho_truck = (
    combined['outbound_truck_wait_min'] / 30.0
).fillna(0).clip(0, 1 - eps).astype('float32')

# Store for interaction (will drop later)
combined['rho_pack_calc'] = rho_pack
combined['rho_robot_calc'] = rho_robot
combined['rho_charge_calc'] = rho_charge
combined['rho_truck_calc'] = rho_truck

# M/M/1 explosion term: rho / (1 - rho)
combined['explosion_pack'] = (rho_pack / (1 - rho_pack)).clip(0, 1000).astype('float32')
combined['explosion_robot'] = (rho_robot / (1 - rho_robot)).clip(0, 1000).astype('float32')
combined['explosion_charge'] = (rho_charge / (1 - rho_charge)).clip(0, 1000).astype('float32')
combined['explosion_truck'] = (rho_truck / (1 - rho_truck)).clip(0, 1000).astype('float32')

# Max bottleneck identification
rho_stack = np.column_stack([rho_pack, rho_robot, rho_charge, rho_truck])
combined['rho_max'] = rho_stack.max(axis=1).astype('float32')
combined['explosion_max'] = (combined['rho_max'] / (1 - combined['rho_max'])).clip(0, 1000).astype('float32')

# Log transform (stabilization)
combined['log_explosion_max'] = np.log1p(combined['explosion_max']).clip(0, 10).astype('float32')
combined['log_explosion_pack'] = np.log1p(combined['explosion_pack']).clip(0, 10).astype('float32')

# Which station is the bottleneck (categorical)
combined['bottleneck_idx'] = rho_stack.argmax(axis=1).astype('int8')

# Number of saturated stations (rho > 0.9)
combined['n_saturated'] = (
    (rho_pack > 0.9).astype(int) +
    (rho_robot > 0.9).astype(int) +
    (rho_charge > 0.9).astype(int) +
    (rho_truck > 0.9).astype(int)
).astype('int8')

# Number of near-saturated stations (rho > 0.8)
combined['n_near_saturated'] = (
    (rho_pack > 0.8).astype(int) +
    (rho_robot > 0.8).astype(int) +
    (rho_charge > 0.8).astype(int) +
    (rho_truck > 0.8).astype(int)
).astype('int8')

# Sum of all explosions (total system pressure)
combined['explosion_sum'] = (
    combined['explosion_pack'] + combined['explosion_robot'] +
    combined['explosion_charge'] + combined['explosion_truck']
).clip(0, 4000).astype('float32')

print(f"  Explosion features: 12", flush=True)

# Verify extreme values
for col in ['explosion_max', 'log_explosion_max', 'n_saturated']:
    print(f"  {col}: min={combined[col].min():.2f}, max={combined[col].max():.2f}, mean={combined[col].mean():.2f}", flush=True)

# ---- Category B: Demand-Supply Gap (7) ----
print("\n=== B) Demand-Supply Gap ===", flush=True)

# Demand indicator (weighted by urgency/complexity)
combined['demand_total'] = (
    combined['order_inflow_15m'] *
    (1 + combined['urgent_order_ratio'] * 2 + combined['heavy_item_ratio'])
).fillna(0).astype('float32')

# Supply indicator (effective capacity)
combined['supply_effective'] = (
    combined['robot_active'] *
    combined['robot_utilization'] *
    combined['pack_station_count']
).fillna(0).astype('float32')

# Gap
combined['ds_gap'] = (combined['demand_total'] - combined['supply_effective']).astype('float32')

# Ratio (demand / supply, with protection)
combined['ds_ratio'] = (
    combined['demand_total'] / (combined['supply_effective'] + 1)
).clip(0, 1000).astype('float32')

# Normalized gap
combined['ds_gap_norm'] = (
    combined['ds_gap'] / (combined['supply_effective'] + 1)
).clip(-100, 1000).astype('float32')

# Scenario-level demand-supply max (peak pressure)
combined['s_ds_gap_max'] = combined.groupby('scenario_id')['ds_gap'].transform('max').astype('float32')
combined['s_ds_ratio_max'] = combined.groupby('scenario_id')['ds_ratio'].transform('max').astype('float32')

print(f"  Demand-Supply features: 7", flush=True)

# ---- Category C: Position Features (4) ----
print("\n=== C) Position Features ===", flush=True)

# Position within scenario (0~24)
combined['position_in_scenario'] = combined.groupby('scenario_id').cumcount().astype('int8')
combined['position_norm'] = (combined['position_in_scenario'] / 24.0).astype('float32')

# Position x explosion (late + explosion = worst case)
combined['pos_x_explosion_max'] = (
    combined['position_norm'] * combined['explosion_max']
).clip(0, 1000).astype('float32')

# Position x demand_supply gap
combined['pos_x_ds_gap'] = (
    combined['position_norm'] * combined['ds_gap']
).astype('float32')

print(f"  Position features: 4", flush=True)

# ---- Category D: Layout Hardness Indicators (domain knowledge, no target) ----
print("\n=== D) Layout Hardness Indicators (domain knowledge) ===", flush=True)

# Check which layout columns survived Phase 13s1 drop
available = set(combined.columns)

# l_pack_ratio: pack_station_count / robot_total
if 'pack_station_count' in available and 'robot_total' in available:
    combined['l_pack_ratio'] = (
        combined['pack_station_count'] / (combined['robot_total'] + 1)
    ).astype('float32')

# l_pack_severity: domain rule (pack_station_count only)
if 'pack_station_count' in available:
    combined['l_pack_severity'] = (
        (combined['pack_station_count'] <= 5).astype(int) * 2 +
        (combined['pack_station_count'] <= 4).astype(int) * 3 +
        (combined['pack_station_count'] <= 3).astype(int) * 5
    ).astype('int8')

# l_robot_density: robot_total / floor_area_sqm (may be dropped)
if 'robot_total' in available and 'floor_area_sqm' in available:
    combined['l_robot_density'] = (
        combined['robot_total'] / (combined['floor_area_sqm'] / 1000 + 1)
    ).astype('float32')
elif 'robot_total' in available:
    # Fallback: pack_station_count as proxy
    combined['l_robot_density'] = (
        combined['robot_total'] / (combined['pack_station_count'] + 1)
    ).astype('float32')

# l_effective_capacity: pack x charger / robot
if all(c in available for c in ['pack_station_count', 'charger_count', 'robot_total']):
    combined['l_effective_capacity'] = (
        combined['pack_station_count'] * combined['charger_count'] / (combined['robot_total'] + 1)
    ).astype('float32')

# l_capacity_demand_ratio: pack x charger / scenario order_inflow mean
if all(c in available for c in ['pack_station_count', 'charger_count']):
    layout_inflow_mean = combined.groupby('layout_id')['order_inflow_15m'].transform('mean')
    combined['l_capacity_demand_ratio'] = (
        combined['pack_station_count'] * combined['charger_count'] / (layout_inflow_mean + 1)
    ).astype('float32')

# l_hardness_score: composite (only if enough components exist)
hardness_cols_for_score = [c for c in [
    'l_pack_ratio', 'l_robot_density', 'l_effective_capacity', 'l_pack_severity'
] if c in combined.columns]

if len(hardness_cols_for_score) >= 2:
    score = pd.Series(0.0, index=combined.index, dtype='float32')
    if 'l_pack_ratio' in combined.columns:
        score -= combined['l_pack_ratio'] * 10
    if 'l_robot_density' in combined.columns:
        score += combined['l_robot_density'] * 0.1
    if 'l_effective_capacity' in combined.columns:
        score -= combined['l_effective_capacity'] * 5
    if 'l_pack_severity' in combined.columns:
        score += combined['l_pack_severity'] * 2
    combined['l_hardness_score'] = score.astype('float32')

# Report actually created features
hardness_feats_created = [c for c in combined.columns if c.startswith('l_') and c in [
    'l_pack_ratio', 'l_pack_severity', 'l_robot_density',
    'l_effective_capacity', 'l_capacity_demand_ratio', 'l_hardness_score'
]]
print(f"  Layout hardness features created: {len(hardness_feats_created)}", flush=True)
print(f"  Features: {hardness_feats_created}", flush=True)

# ---- Category E: Coefficient of Variation Squared (12) ----
print("\n=== E) CV^2 Features (M/G/1 formula) ===", flush=True)

combined = combined.sort_values(['scenario_id', 'implicit_timeslot']).reset_index(drop=True)

CV_TARGETS = [
    'pack_utilization', 'robot_active', 'congestion_score',
    'order_inflow_15m', 'charge_queue_length', 'max_zone_density',
]
CV_TARGETS = [c for c in CV_TARGETS if c in combined.columns]

for col in CV_TARGETS:
    s_std = combined.groupby('scenario_id')[col].transform('std').fillna(0)
    s_mean = combined.groupby('scenario_id')[col].transform('mean').fillna(0)

    # CV = std / mean
    cv = (s_std / (s_mean.abs() + 0.01)).clip(0, 100).astype('float32')
    combined[f'cv_{col}'] = cv

    # CV^2 (M/G/1 waiting time formula core)
    combined[f'cv_sq_{col}'] = (cv ** 2).clip(0, 10000).astype('float32')

n_cv_feats = 2 * len(CV_TARGETS)
print(f"  CV^2 features: {n_cv_feats}", flush=True)


# ##############################################################
# Part 3: NaN Cleanup + Verification
# ##############################################################
print("\n=== NaN Cleanup ===", flush=True)

new_phase17_feats = [
    c for c in combined.columns
    if c.startswith((
        'explosion_', 'rho_max', 'log_explosion', 'bottleneck_idx',
        'n_saturated', 'n_near_saturated',
        'demand_total', 'supply_effective', 'ds_gap', 'ds_ratio', 's_ds_',
        'position_in_scenario', 'position_norm', 'pos_x_',
        'l_',  # all l_* layout hardness features
        'cv_'
    ))
    and c not in phase16_feature_cols  # exclude Phase 16 base overlaps
]
# Remove calc intermediates
new_phase17_feats = [c for c in new_phase17_feats if not c.endswith('_calc')]
# Drop calc intermediates
for c in ['rho_pack_calc', 'rho_robot_calc', 'rho_charge_calc', 'rho_truck_calc']:
    if c in combined.columns:
        combined.drop(columns=[c], inplace=True)

print(f"Total new Phase 17 features: {len(new_phase17_feats)}", flush=True)

for col in new_phase17_feats:
    if combined[col].dtype in [np.float32, np.float64]:
        combined[col] = np.nan_to_num(
            combined[col].values.astype('float32'),
            nan=0.0, posinf=1000.0, neginf=-1000.0
        )
print("NaN cleanup done", flush=True)

# Verify: no NaN/inf
for col in new_phase17_feats:
    n_nan = pd.isna(combined[col]).sum()
    n_inf = np.isinf(combined[col].astype('float64')).sum() if combined[col].dtype != 'int8' else 0
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING {col}: nan={n_nan}, inf={n_inf}", flush=True)


# ##############################################################
# Part 4: Feature Selection (Phase 17 new features only)
# ##############################################################
print("\n=== Phase 17 Feature Selection ===", flush=True)

combined = combined.sort_values('_original_idx').reset_index(drop=True)

# Phase 16 feature_cols + new Phase 17 features
temp_feature_cols = phase16_feature_cols + new_phase17_feats
print(f"Temp total: {len(temp_feature_cols)}", flush=True)

X_tmp = combined.loc[combined['_is_train'] == 1, temp_feature_cols].fillna(0).astype('float32')

lgb_selector = lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=8,
    min_child_samples=50,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    deterministic=True,
    force_col_wise=True,
    n_jobs=1,
    verbosity=-1,
)

tr_idx_sel, va_idx_sel = next(gkf.split(X_tmp, y_train, groups=layout_ids_train))

lgb_selector.fit(
    X_tmp.iloc[tr_idx_sel], y_train[tr_idx_sel],
    eval_set=[(X_tmp.iloc[va_idx_sel], y_train[va_idx_sel])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)

imp_df = pd.DataFrame({
    'feature': temp_feature_cols,
    'importance': lgb_selector.feature_importances_,
}).sort_values('importance', ascending=False).reset_index(drop=True)

# Phase 17 new features only
new_imp = imp_df[imp_df['feature'].isin(new_phase17_feats)].copy()
new_imp['rank'] = np.arange(1, len(new_imp) + 1)

print(f"\n=== Top 30 new Phase 17 features ===", flush=True)
print(new_imp.head(30).to_string(index=False), flush=True)

# Category breakdown
print(f"\n=== Category breakdown (new features) ===", flush=True)
for prefix_group, name in [
    (('explosion_', 'log_explosion_', 'n_satur', 'n_near_satur', 'bottleneck_idx'), 'A. Explosion'),
    (('demand_', 'supply_', 'ds_', 's_ds_'), 'B. Demand-Supply'),
    (('position_', 'pos_x_'), 'C. Position'),
    (('l_pack_ratio', 'l_pack_severity', 'l_robot_density', 'l_effective_capacity', 'l_capacity_demand_ratio', 'l_hardness_score'), 'D. Layout Hardness'),
    (('cv_',), 'E. CV^2'),
]:
    feats_in_cat = [f for f in new_phase17_feats if f.startswith(prefix_group)]
    top30 = new_imp.head(30)['feature'].tolist()
    in_top30 = sum(1 for f in feats_in_cat if f in top30)
    total = len(feats_in_cat)
    print(f"  {name}: {in_top30}/{total} in top 30", flush=True)

# Select top 30
TOP_N_NEW = 30
selected_new = new_imp.head(TOP_N_NEW)['feature'].tolist()
dropped_new = [f for f in new_phase17_feats if f not in selected_new]

# Force-include best layout hardness feature
forced_feat = None
if 'l_hardness_score' in new_phase17_feats:
    forced_feat = 'l_hardness_score'
elif 'l_pack_severity' in new_phase17_feats:
    forced_feat = 'l_pack_severity'

if forced_feat and forced_feat not in selected_new:
    selected_new.append(forced_feat)
    if forced_feat in dropped_new:
        dropped_new.remove(forced_feat)
    print(f"\n  {forced_feat} force-included", flush=True)

print(f"\nSelection: kept {len(selected_new)}, dropped {len(dropped_new)}", flush=True)

combined.drop(columns=[c for c in dropped_new if c in combined.columns], inplace=True)

final_feature_cols = phase16_feature_cols + selected_new
print(f"\n=== Final total: {len(final_feature_cols)} features ===", flush=True)

new_imp.to_csv('output/phase18_feature_importance.csv', index=False)
del X_tmp, lgb_selector; gc.collect()

# Prepare train/test arrays for two-stage
feature_cols = final_feature_cols
combined = combined.sort_values('_original_idx').reset_index(drop=True)
train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()
del combined; gc.collect()

X_train = train_fe[feature_cols].fillna(0).astype('float32')
X_test = test_fe[feature_cols].fillna(0).astype('float32')
y = train_fe['avg_delay_minutes_next_30m'].values
sample_sub = pd.read_csv('data/sample_submission.csv')
assert (test_fe['ID'].values == sample_sub['ID'].values).all()
print(f"Features: {len(feature_cols)}, ID verified", flush=True)


# ##############################################################
# Part 1: Phase 16 Ensemble OOF Reconstruction (Stage 2A)
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 1: Phase 16 Ensemble Reconstruction ===", flush=True)
print("=" * 60, flush=True)

P16_MODELS = ['lgb_raw', 'lgb_huber', 'lgb_sqrt', 'xgb', 'cat_log1p', 'cat_raw', 'mlp']

p16_oofs = {}
p16_tests = {}
for name in P16_MODELS:
    path = f'output/ckpt_phase16_{name}.pkl'
    if not os.path.exists(path):
        drive_path = f'/content/drive/MyDrive/dacon_ckpt/ckpt_phase16_{name}.pkl'
        if os.path.exists(drive_path):
            shutil.copy(drive_path, path)
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    if 'oof' not in ckpt or len(ckpt['oof']) != len(y):
        raise RuntimeError(f'Phase 16 checkpoint shape mismatch: {name}')
    p16_oofs[name] = ckpt['oof']
    p16_tests[name] = ckpt['test']

# Nelder-Mead weight reconstruction (normalized)
def nm_obj(w, oof_m, yt):
    w = w / (w.sum() + 1e-12)
    return np.abs((oof_m * w).sum(axis=1) - yt).mean()

p16_oof_mat = np.column_stack([p16_oofs[n] for n in P16_MODELS])
p16_test_mat = np.column_stack([p16_tests[n] for n in P16_MODELS])

x0 = np.ones(len(P16_MODELS)) / len(P16_MODELS)
p16_result = minimize(nm_obj, x0, args=(p16_oof_mat, y),
                      method='Nelder-Mead',
                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})
p16_w = p16_result.x / (p16_result.x.sum() + 1e-12)

oof_normal = (p16_oof_mat * p16_w).sum(axis=1)  # Stage 2A OOF
test_normal = (p16_test_mat * p16_w).sum(axis=1)  # Stage 2A test

p16_cv = np.abs(oof_normal - y).mean()
print(f"  Phase 16 reconstructed CV: {p16_cv:.4f} (expected ~8.4403)", flush=True)

print(f"\n  P16 weights:", flush=True)
for n, w in zip(P16_MODELS, p16_w):
    print(f"    {n}: {w:.4f}", flush=True)


# ##############################################################
# Part 2: Stage 1 — Extreme Classifier (OOF)
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 2: Stage 1 — Extreme Classifier ===", flush=True)
print("=" * 60, flush=True)

from sklearn.metrics import roc_auc_score

# Threshold experiments
THRESHOLDS = [50, 80, 100]

# CV definition (shared across all stages)
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv_splitter.split(np.arange(len(y)), y_binned, groups=train_fe['layout_id']))

best_threshold = None
best_combined_cv = 999
best_oof_proba = None
best_test_proba = None
best_auc = 0

for T_candidate in THRESHOLDS:
    print(f"\n  --- Threshold T={T_candidate} ---", flush=True)

    # Binary target
    y_extreme_cand = (y > T_candidate).astype(int)
    n_extreme_cand = y_extreme_cand.sum()
    print(f"  Extreme samples: {n_extreme_cand} ({n_extreme_cand/len(y)*100:.2f}%)", flush=True)

    if n_extreme_cand < 500:
        print(f"  Too few extreme samples, skip", flush=True)
        continue

    # OOF classifier
    oof_proba_cand = np.zeros(len(y), dtype='float32')
    test_proba_cand = np.zeros(len(X_test), dtype='float32')

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=8,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=1.0,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=5,
            class_weight='balanced',
            verbosity=-1,
            random_state=42,
        )

        clf.fit(
            X_train.iloc[tr_idx], y_extreme_cand[tr_idx],
            eval_set=[(X_train.iloc[va_idx], y_extreme_cand[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        oof_proba_cand[va_idx] = clf.predict_proba(X_train.iloc[va_idx])[:, 1].astype('float32')
        test_proba_cand += clf.predict_proba(X_test)[:, 1].astype('float32') / 5

    auc_cand = roc_auc_score(y_extreme_cand, oof_proba_cand)
    print(f"  Classifier AUC: {auc_cand:.4f}", flush=True)

    # Quick estimation with placeholder (extreme mean)
    extreme_mean = y[y_extreme_cand == 1].mean()
    temp_pred = (1 - oof_proba_cand) * oof_normal + oof_proba_cand * extreme_mean
    temp_cv = np.abs(temp_pred - y).mean()
    print(f"  Temp combined CV (placeholder): {temp_cv:.4f}", flush=True)

    if temp_cv < best_combined_cv:
        best_combined_cv = temp_cv
        best_threshold = T_candidate
        best_oof_proba = oof_proba_cand.copy()
        best_test_proba = test_proba_cand.copy()
        best_auc = auc_cand

print(f"\n  Best threshold: T={best_threshold}", flush=True)
print(f"  Best AUC: {best_auc:.4f}", flush=True)

T = best_threshold
y_extreme = (y > T).astype(int)
oof_proba = best_oof_proba
test_proba = best_test_proba


# ##############################################################
# Part 3: Stage 2B — Extreme MLP (OOF)
# ##############################################################
print("\n" + "=" * 60, flush=True)
print(f"=== Part 3: Stage 2B — Extreme MLP (T={T}) ===", flush=True)
print("=" * 60, flush=True)

import tensorflow as tf
from tensorflow.keras import layers, callbacks as keras_callbacks

def build_extreme_mlp(n_features):
    """Small MLP for extreme samples (3K~12K)"""
    model = tf.keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation='gelu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='gelu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='gelu'),
        layers.Dropout(0.2),
        layers.Dense(1)  # Linear output for extrapolation
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mae'
    )
    return model

# Extreme samples mask
extreme_mask = (y > T)
n_extreme = extreme_mask.sum()
print(f"  Extreme samples: {n_extreme}", flush=True)

# Feature selection for extreme MLP (top 150 by importance)
EXTREME_N_FEATURES = min(150, len(feature_cols))
print(f"  Using top {EXTREME_N_FEATURES} features for extreme MLP", flush=True)
extreme_feature_cols = feature_cols[:EXTREME_N_FEATURES]

# Prepare raw arrays
X_train_raw = np.nan_to_num(X_train[extreme_feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')
X_test_raw = np.nan_to_num(X_test[extreme_feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')

# OOF extreme prediction
oof_extreme = np.zeros(len(y), dtype='float32')
test_extreme = np.zeros(len(X_test), dtype='float32')

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    print(f"\n  Extreme MLP Fold {fold_idx+1}/5...", flush=True)

    # This fold's train extreme samples only
    tr_extreme_mask = extreme_mask[tr_idx]
    tr_extreme_idx = tr_idx[tr_extreme_mask]
    n_fold_extreme = len(tr_extreme_idx)
    print(f"    Train extreme: {n_fold_extreme}", flush=True)

    if n_fold_extreme < 100:
        print(f"    Too few, skip fold (use normal pred)", flush=True)
        oof_extreme[va_idx] = oof_normal[va_idx]
        continue

    # Fold-isolated scaler
    scaler = StandardScaler()
    X_tr_scaled = np.clip(scaler.fit_transform(X_train_raw[tr_extreme_idx]), -5, 5).astype('float32')

    X_va_scaled = np.clip(scaler.transform(X_train_raw[va_idx]), -5, 5).astype('float32')
    X_te_scaled = np.clip(scaler.transform(X_test_raw), -5, 5).astype('float32')

    y_tr_extreme = y[tr_extreme_idx].astype('float32')

    # Validation for early stopping: fold's extreme samples
    va_extreme_mask = extreme_mask[va_idx]
    if va_extreme_mask.sum() > 10:
        X_va_ext = X_va_scaled[va_extreme_mask]
        y_va_ext = y[va_idx[va_extreme_mask]].astype('float32')
        val_data = (X_va_ext, y_va_ext)
    else:
        val_data = None

    tf.random.set_seed(42); np.random.seed(42)
    model = build_extreme_mlp(len(extreme_feature_cols))

    model.fit(
        X_tr_scaled, y_tr_extreme,
        validation_data=val_data,
        epochs=100,
        batch_size=min(256, n_fold_extreme),
        callbacks=[
            keras_callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
            keras_callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5, verbose=0),
        ],
        verbose=0,
    )

    # Predict
    oof_extreme[va_idx] = model.predict(X_va_scaled, verbose=0).flatten().clip(0, None).astype('float32')
    test_extreme += model.predict(X_te_scaled, verbose=0).flatten().clip(0, None).astype('float32') / 5

    # Fold results
    va_mae = np.abs(oof_extreme[va_idx] - y[va_idx]).mean()
    va_extreme_mae = np.abs(oof_extreme[va_idx[va_extreme_mask]] - y[va_idx[va_extreme_mask]]).mean() if va_extreme_mask.sum() > 0 else float('nan')
    print(f"    Fold {fold_idx+1} all-sample MAE: {va_mae:.4f}", flush=True)
    print(f"    Fold {fold_idx+1} extreme-only MAE: {va_extreme_mae:.4f}", flush=True)

    del model; tf.keras.backend.clear_session(); gc.collect()

ext_cv = np.abs(oof_extreme - y).mean()
print(f"\n  Extreme MLP full CV: {ext_cv:.4f}", flush=True)

# Extreme-only MAE (key metric)
ext_only_mae = np.abs(oof_extreme[extreme_mask] - y[extreme_mask]).mean()
print(f"  Extreme-only MAE: {ext_only_mae:.4f} (Phase 16: 40.92)", flush=True)

# Checkpoint
save_ckpt('output/ckpt_phase18_extreme_mlp.pkl', {
    'oof': oof_extreme,
    'test': test_extreme,
    'threshold': T,
})


# ##############################################################
# Part 4: Combine — Probability-weighted Blend
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 4: Combine (Probability-weighted) ===", flush=True)
print("=" * 60, flush=True)

# Method A: Soft blend
oof_combined_soft = (1 - oof_proba) * oof_normal + oof_proba * oof_extreme
test_combined_soft = (1 - test_proba) * test_normal + test_proba * test_extreme
cv_soft = np.abs(oof_combined_soft - y).mean()
print(f"  Soft blend CV: {cv_soft:.4f}", flush=True)

# Method B: Hard threshold search
print("\n  Hard threshold search:", flush=True)
best_hard_cv = 999
best_hard_cutoff = 0.5

for cutoff in [0.3, 0.4, 0.5, 0.6, 0.7]:
    oof_hard = np.where(oof_proba > cutoff, oof_extreme, oof_normal)
    cv_hard = np.abs(oof_hard - y).mean()
    marker = ""
    if cv_hard < best_hard_cv:
        best_hard_cv = cv_hard
        best_hard_cutoff = cutoff
        marker = " *"
    print(f"    cutoff={cutoff:.1f}: CV {cv_hard:.4f}{marker}", flush=True)

# Method C: Adjustment ratio
print("\n  Adjustment ratio search:", flush=True)
best_alpha_cv = 999
best_alpha = 1.0

for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    adjustment = alpha * oof_proba * (oof_extreme - oof_normal)
    oof_adj = oof_normal + adjustment
    cv_adj = np.abs(oof_adj - y).mean()
    marker = ""
    if cv_adj < best_alpha_cv:
        best_alpha_cv = cv_adj
        best_alpha = alpha
        marker = " *"
    print(f"    alpha={alpha:.1f}: CV {cv_adj:.4f}{marker}", flush=True)

# Select best method
methods = {
    'soft': cv_soft,
    'hard': best_hard_cv,
    'adjustment': best_alpha_cv,
    'normal_only': p16_cv,
}
best_method = min(methods, key=methods.get)
print(f"\n  Best method: {best_method} (CV {methods[best_method]:.4f})", flush=True)
print(f"  Phase 16 baseline: {p16_cv:.4f}", flush=True)
print(f"  Improvement: {p16_cv - methods[best_method]:+.4f}", flush=True)

# Generate prediction with best method
if best_method == 'soft':
    final_oof = oof_combined_soft
    final_test = test_combined_soft
elif best_method == 'hard':
    final_oof = np.where(oof_proba > best_hard_cutoff, oof_extreme, oof_normal)
    final_test = np.where(test_proba > best_hard_cutoff, test_extreme, test_normal)
elif best_method == 'adjustment':
    final_oof = oof_normal + best_alpha * oof_proba * (oof_extreme - oof_normal)
    final_test = test_normal + best_alpha * test_proba * (test_extreme - test_normal)
else:
    final_oof = oof_normal.copy()
    final_test = test_normal.copy()

final_oof = np.clip(final_oof, 0, None)
final_test = np.clip(final_test, 0, 1000)


# ##############################################################
# Part 5: Isotonic Calibration (Nested OOF)
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 5: Isotonic Calibration ===", flush=True)
print("=" * 60, flush=True)

from sklearn.isotonic import IsotonicRegression

# Nested CV isotonic (same folds)
oof_calibrated = np.zeros(len(y), dtype='float32')

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso.fit(final_oof[tr_idx], y[tr_idx])

    oof_calibrated[va_idx] = iso.predict(final_oof[va_idx]).astype('float32')

    fold_before = np.abs(final_oof[va_idx] - y[va_idx]).mean()
    fold_after = np.abs(oof_calibrated[va_idx] - y[va_idx]).mean()
    print(f"  Fold {fold_idx+1}: before {fold_before:.4f} -> after {fold_after:.4f} "
          f"({fold_before - fold_after:+.4f})", flush=True)

cv_calibrated = np.abs(oof_calibrated - y).mean()
print(f"\n  Calibrated CV: {cv_calibrated:.4f}", flush=True)
print(f"  Pre-calibration: {methods[best_method]:.4f}", flush=True)
print(f"  Phase 16 baseline: {p16_cv:.4f}", flush=True)

# Test isotonic (fit on full train)
iso_final = IsotonicRegression(out_of_bounds='clip', increasing=True)
iso_final.fit(final_oof, y)
test_calibrated = iso_final.predict(final_test).astype('float32')

# Use calibration only if it improves
use_calibration = cv_calibrated < methods[best_method] - 0.001

if use_calibration:
    submission_pred = test_calibrated
    submission_cv = cv_calibrated
    print(f"\n  Isotonic calibration applied", flush=True)
else:
    submission_pred = final_test
    submission_cv = methods[best_method]
    print(f"\n  Isotonic calibration skipped (no improvement)", flush=True)


# ##############################################################
# Part 6: Submission + Validation
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 6: Submission + Validation ===", flush=True)
print("=" * 60, flush=True)

submission_pred = np.clip(submission_pred, 0, 1000)

sub = pd.DataFrame({
    'ID': test_fe['ID'].values,
    'avg_delay_minutes_next_30m': submission_pred.astype('float32')
})
sub = sub.set_index('ID').reindex(sample_sub['ID']).reset_index()
assert (sub['ID'].values == sample_sub['ID'].values).all()

sub.to_csv('output/submission_phase18.csv', index=False)
shutil.copy('output/submission_phase18.csv',
            os.path.join(DRIVE_CKPT_DIR, 'submission_phase18.csv'))

print(f"  submission_phase18.csv saved", flush=True)
print(sub['avg_delay_minutes_next_30m'].describe(), flush=True)

# === Bin 9 Verification ===
print("\n" + "=" * 60, flush=True)
print("=== BIN 9 VERIFICATION (Primary Objective) ===", flush=True)
print("=" * 60, flush=True)

if use_calibration:
    p18_oof_final = oof_calibrated
else:
    p18_oof_final = final_oof

bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
for bin_idx in range(10):
    mask = (bins == bin_idx)
    p16_mae = np.abs(oof_normal[mask] - y[mask]).mean()
    p18_mae = np.abs(p18_oof_final[mask] - y[mask]).mean()

    target_mean = y[mask].mean()
    improvement = p16_mae - p18_mae
    marker = " ***" if bin_idx == 9 and improvement > 5 else (
             " **" if improvement > 1 else (
             " *" if improvement > 0.1 else ""))

    print(f"  Bin {bin_idx}: target_mean={target_mean:6.1f}, "
          f"P16 MAE={p16_mae:6.2f}, P18 MAE={p18_mae:6.2f}, "
          f"d={improvement:+.2f}{marker}", flush=True)

# Hard layouts verification
HARD_TOP5 = ['WH_051', 'WH_073', 'WH_217', 'WH_049', 'WH_098']
hard_mask = train_fe['layout_id'].isin(HARD_TOP5).values
p16_hard = np.abs(oof_normal[hard_mask] - y[hard_mask]).mean()
p18_hard = np.abs(p18_oof_final[hard_mask] - y[hard_mask]).mean()
print(f"\n  Hard top5: P16={p16_hard:.2f}, P18={p18_hard:.2f}, d={p16_hard-p18_hard:+.2f}", flush=True)

# Summary
print(f"\n{'='*60}", flush=True)
print(f"=== PHASE 18 SUMMARY ===", flush=True)
print(f"{'='*60}", flush=True)
print(f"Threshold: T={T}", flush=True)
print(f"Classifier AUC: {best_auc:.4f}", flush=True)
print(f"Combine method: {best_method}", flush=True)
if best_method == 'hard':
    print(f"  Cutoff: {best_hard_cutoff}", flush=True)
elif best_method == 'adjustment':
    print(f"  Alpha: {best_alpha}", flush=True)
print(f"Isotonic applied: {use_calibration}", flush=True)
print(f"", flush=True)
print(f"Phase 16 CV:      {p16_cv:.4f}", flush=True)
print(f"Phase 18 CV:      {submission_cv:.4f}", flush=True)
print(f"Improvement:      {p16_cv - submission_cv:+.4f}", flush=True)
print(f"Bin 9 MAE P16:    {np.abs(oof_normal[bins==bins.max()] - y[bins==bins.max()]).mean():.2f}", flush=True)
print(f"Bin 9 MAE P18:    {np.abs(p18_oof_final[bins==bins.max()] - y[bins==bins.max()]).mean():.2f}", flush=True)

print("\n=== Phase 18 Complete ===", flush=True)
