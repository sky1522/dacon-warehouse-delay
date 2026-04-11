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

PIPELINE_VERSION = "phase21c_v1"


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
            # Metadata validation
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

y_train = combined.loc[combined['_is_train'] == 1, 'avg_delay_minutes_next_30m'].values
layout_ids_train = combined.loc[combined['_is_train'] == 1, 'layout_id'].values
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
# Part 2: 2nd-order Features (NEW)
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 2: 2nd-order Features ===", flush=True)
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

# ##############################################################
# Part 3: Feature Selection (2nd-order)
# ##############################################################
print("\n=== Part 3: 2nd-order Feature Selection ===", flush=True)
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
final_feature_cols = feature_cols_p15 + selected_2nd
print(f"\n  Final total features: {len(final_feature_cols)}", flush=True)
new_imp2.to_csv('output/phase21c_feature_importance.csv', index=False)
del X_tmp2, sel2; gc.collect()

# ##############################################################
# Part 4: Training (8 models)
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Part 4: Training ===", flush=True)
print("=" * 60, flush=True)

train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()
del combined; gc.collect()

feature_cols = final_feature_cols
X = train_fe[feature_cols]
y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y); y_sqrt = np.sqrt(y)
groups = train_fe['layout_id']
time_idx = train_fe['implicit_timeslot'].values.astype(np.float32)
X_test = test_fe[feature_cols]
assert (test_fe['ID'].values == sample_sub['ID'].values).all()
print(f"Features: {len(feature_cols)}, ID verified", flush=True)

y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(sgkf.split(X, y_binned, groups=groups))

def build_sample_weight(y_arr, time_arr):
    w = np.ones(len(y_arr), dtype=np.float32)
    q90, q95, q99 = np.nanquantile(y_arr, [0.90, 0.95, 0.99])
    w += 0.15 * (y_arr >= q90).astype(np.float32)
    w += 0.30 * (y_arr >= q95).astype(np.float32)
    w += 0.60 * (y_arr >= q99).astype(np.float32)
    if time_arr is not None: w += 0.08 * (time_arr / 24.0).astype(np.float32)
    return w
sample_w = build_sample_weight(y.values, time_idx)

# NN preprocessing: raw arrays for fold-internal scaling (no leakage)
X_train_nn_raw = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_test_nn_raw = np.nan_to_num(X_test.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y_log_nn = y_log.values.astype(np.float32)

oof_preds = {}; test_preds = {}; cv_maes = {}

# --- Model 1: LGB raw+MAE ---
print("\n  [1] LGB raw+MAE", flush=True)
ckpt_path = 'output/ckpt_phase21c_lgb_raw.pkl'; ckpt = load_ckpt(ckpt_path, expected_features=feature_cols)
if ckpt: oof_preds['lgb_raw'] = ckpt['oof']; test_preds['lgb_raw'] = ckpt['test']; cv_maes['lgb_raw'] = ckpt['cv_mae']; m1_models = []
else:
    o, t, m1_models = np.zeros(len(X), dtype=np.float32), np.zeros(len(X_test), dtype=np.float32), []
    for fi, (tri, vai) in enumerate(folds):
        m = lgb.LGBMRegressor(objective='mae', n_estimators=2000, learning_rate=0.0129, num_leaves=185, max_depth=9, min_child_samples=80, reg_alpha=0.0574, reg_lambda=0.0042, feature_fraction=0.6005, bagging_fraction=0.7663, bagging_freq=1, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tri], y.iloc[tri], sample_weight=sample_w[tri], eval_set=[(X.iloc[vai], y.iloc[vai])], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        p = np.clip(m.predict(X.iloc[vai]), 0, None).astype(np.float32); o[vai] = p; t += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5; m1_models.append(m)
        print(f"    F{fi+1} MAE: {mean_absolute_error(y.iloc[vai], p):.4f}", flush=True)
    cv_maes['lgb_raw'] = mean_absolute_error(y, o); oof_preds['lgb_raw'] = o; test_preds['lgb_raw'] = t
    save_ckpt(ckpt_path, {'oof': o, 'test': t, 'cv_mae': cv_maes['lgb_raw']}, feature_cols=feature_cols)
print(f"  CV: {cv_maes['lgb_raw']:.4f}", flush=True)

# --- Model 2: LGB log1p+Huber ---
print("\n  [2] LGB log1p+Huber", flush=True)
ckpt_path = 'output/ckpt_phase21c_lgb_huber.pkl'; ckpt = load_ckpt(ckpt_path, expected_features=feature_cols)
if ckpt: oof_preds['lgb_huber'] = ckpt['oof']; test_preds['lgb_huber'] = ckpt['test']; cv_maes['lgb_huber'] = ckpt['cv_mae']
else:
    o, t = np.zeros(len(X), dtype=np.float32), np.zeros(len(X_test), dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        m = lgb.LGBMRegressor(objective='huber', huber_delta=0.9, n_estimators=2000, learning_rate=0.03, num_leaves=128, min_child_samples=60, subsample=0.9, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tri], y_log.iloc[tri], sample_weight=sample_w[tri], eval_set=[(X.iloc[vai], y_log.iloc[vai])], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        p = np.clip(np.expm1(m.predict(X.iloc[vai])), 0, None).astype(np.float32); o[vai] = p; t += np.clip(np.expm1(m.predict(X_test)), 0, None).astype(np.float32) / 5
        print(f"    F{fi+1} MAE: {mean_absolute_error(y.iloc[vai], p):.4f}", flush=True)
    cv_maes['lgb_huber'] = mean_absolute_error(y, o); oof_preds['lgb_huber'] = o; test_preds['lgb_huber'] = t
    save_ckpt(ckpt_path, {'oof': o, 'test': t, 'cv_mae': cv_maes['lgb_huber']}, feature_cols=feature_cols)
print(f"  CV: {cv_maes['lgb_huber']:.4f}", flush=True)

# --- Model 3: LGB sqrt ---
print("\n  [3] LGB sqrt", flush=True)
ckpt_path = 'output/ckpt_phase21c_lgb_sqrt.pkl'; ckpt = load_ckpt(ckpt_path, expected_features=feature_cols)
if ckpt: oof_preds['lgb_sqrt'] = ckpt['oof']; test_preds['lgb_sqrt'] = ckpt['test']; cv_maes['lgb_sqrt'] = ckpt['cv_mae']
else:
    o, t = np.zeros(len(X), dtype=np.float32), np.zeros(len(X_test), dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        m = lgb.LGBMRegressor(objective='mae', n_estimators=2000, learning_rate=0.03, num_leaves=96, min_child_samples=80, subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tri], y_sqrt.iloc[tri], sample_weight=sample_w[tri], eval_set=[(X.iloc[vai], y_sqrt.iloc[vai])], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        p = np.clip(m.predict(X.iloc[vai]) ** 2, 0, None).astype(np.float32); o[vai] = p; t += np.clip(m.predict(X_test) ** 2, 0, None).astype(np.float32) / 5
        print(f"    F{fi+1} MAE: {mean_absolute_error(y.iloc[vai], p):.4f}", flush=True)
    cv_maes['lgb_sqrt'] = mean_absolute_error(y, o); oof_preds['lgb_sqrt'] = o; test_preds['lgb_sqrt'] = t
    save_ckpt(ckpt_path, {'oof': o, 'test': t, 'cv_mae': cv_maes['lgb_sqrt']}, feature_cols=feature_cols)
print(f"  CV: {cv_maes['lgb_sqrt']:.4f}", flush=True)

# --- Model 4: XGB raw ---
print("\n  [4] XGB raw", flush=True)
ckpt_path = 'output/ckpt_phase21c_xgb.pkl'; ckpt = load_ckpt(ckpt_path, expected_features=feature_cols)
if ckpt: oof_preds['xgb'] = ckpt['oof']; test_preds['xgb'] = ckpt['test']; cv_maes['xgb'] = ckpt['cv_mae']
else:
    o, t = np.zeros(len(X), dtype=np.float32), np.zeros(len(X_test), dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8, min_child_weight=6, subsample=0.9, colsample_bytree=0.85, reg_lambda=1.5, reg_alpha=0.05, objective='reg:absoluteerror', eval_metric='mae', tree_method='hist', random_state=42, verbosity=0, early_stopping_rounds=100)
        m.fit(X.iloc[tri], y.iloc[tri], sample_weight=sample_w[tri], eval_set=[(X.iloc[vai], y.iloc[vai])], verbose=False)
        p = np.clip(m.predict(X.iloc[vai]), 0, None).astype(np.float32); o[vai] = p; t += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
        print(f"    F{fi+1} MAE: {mean_absolute_error(y.iloc[vai], p):.4f}", flush=True)
    cv_maes['xgb'] = mean_absolute_error(y, o); oof_preds['xgb'] = o; test_preds['xgb'] = t
    save_ckpt(ckpt_path, {'oof': o, 'test': t, 'cv_mae': cv_maes['xgb']}, feature_cols=feature_cols)
print(f"  CV: {cv_maes['xgb']:.4f}", flush=True)

# --- Model 5: Cat log1p ---
print("\n  [5] Cat log1p", flush=True)
ckpt_path = 'output/ckpt_phase21c_cat_log1p.pkl'; ckpt = load_ckpt(ckpt_path, expected_features=feature_cols)
if ckpt: oof_preds['cat_log1p'] = ckpt['oof']; test_preds['cat_log1p'] = ckpt['test']; cv_maes['cat_log1p'] = ckpt['cv_mae']
else:
    o, t = np.zeros(len(X), dtype=np.float32), np.zeros(len(X_test), dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        m = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=5.0, subsample=0.9, loss_function='MAE', random_seed=42, verbose=0)
        m.fit(Pool(X.iloc[tri], y_log.iloc[tri], weight=sample_w[tri]), eval_set=Pool(X.iloc[vai], y_log.iloc[vai]), early_stopping_rounds=100)
        p = np.clip(np.expm1(m.predict(X.iloc[vai])), 0, None).astype(np.float32); o[vai] = p; t += np.clip(np.expm1(m.predict(X_test)), 0, None).astype(np.float32) / 5
        print(f"    F{fi+1} MAE: {mean_absolute_error(y.iloc[vai], p):.4f}", flush=True)
    cv_maes['cat_log1p'] = mean_absolute_error(y, o); oof_preds['cat_log1p'] = o; test_preds['cat_log1p'] = t
    save_ckpt(ckpt_path, {'oof': o, 'test': t, 'cv_mae': cv_maes['cat_log1p']}, feature_cols=feature_cols)
print(f"  CV: {cv_maes['cat_log1p']:.4f}", flush=True)

# --- Model 6: Cat raw ---
print("\n  [6] Cat raw", flush=True)
ckpt_path = 'output/ckpt_phase21c_cat_raw.pkl'; ckpt = load_ckpt(ckpt_path, expected_features=feature_cols)
if ckpt: oof_preds['cat_raw'] = ckpt['oof']; test_preds['cat_raw'] = ckpt['test']; cv_maes['cat_raw'] = ckpt['cv_mae']
else:
    o, t = np.zeros(len(X), dtype=np.float32), np.zeros(len(X_test), dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        m = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, subsample=0.85, loss_function='MAE', random_seed=42, verbose=0)
        m.fit(Pool(X.iloc[tri], y.iloc[tri], weight=sample_w[tri]), eval_set=Pool(X.iloc[vai], y.iloc[vai]), early_stopping_rounds=100)
        p = np.clip(m.predict(X.iloc[vai]), 0, None).astype(np.float32); o[vai] = p; t += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
        print(f"    F{fi+1} MAE: {mean_absolute_error(y.iloc[vai], p):.4f}", flush=True)
    cv_maes['cat_raw'] = mean_absolute_error(y, o); oof_preds['cat_raw'] = o; test_preds['cat_raw'] = t
    save_ckpt(ckpt_path, {'oof': o, 'test': t, 'cv_mae': cv_maes['cat_raw']}, feature_cols=feature_cols)
print(f"  CV: {cv_maes['cat_raw']:.4f}", flush=True)

# --- Model 7: MLP (Phase 21C: loss = (RMSE+MAE)/2) ---
print("\n  [7] Keras MLP (loss = (RMSE+MAE)/2)", flush=True)
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks as keras_callbacks

def rmse_mae_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) + 1e-8)
    return (mae + rmse) / 2

def build_mlp(dim):
    inp = layers.Input(shape=(dim,)); x = layers.Dense(512, activation='relu')(inp); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x); out = layers.Dense(1)(x)
    mdl = Model(inp, out); mdl.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=rmse_mae_loss); return mdl

ckpt_path = 'output/ckpt_phase21c_mlp.pkl'; ckpt = load_ckpt(ckpt_path, expected_features=feature_cols)
if ckpt: oof_preds['mlp'] = ckpt['oof']; test_preds['mlp'] = ckpt['test']; cv_maes['mlp'] = ckpt['cv_mae']
else:
    o, t = np.zeros(len(X), dtype=np.float32), np.zeros(len(X_test), dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        tf.random.set_seed(42); np.random.seed(42)
        # Fold-internal scaling (no leakage)
        fold_scaler = StandardScaler()
        Xtr = np.clip(fold_scaler.fit_transform(X_train_nn_raw[tri]), -5, 5).astype('float32')
        Xva = np.clip(fold_scaler.transform(X_train_nn_raw[vai]), -5, 5).astype('float32')
        Xte = np.clip(fold_scaler.transform(X_test_nn_raw), -5, 5).astype('float32')
        mdl = build_mlp(Xtr.shape[1])
        mdl.fit(Xtr, y_log_nn[tri], validation_data=(Xva, y_log_nn[vai]), epochs=100, batch_size=512, callbacks=[keras_callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True, verbose=0), keras_callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)], verbose=0)
        p = np.clip(np.expm1(mdl.predict(Xva, verbose=0).flatten()), 0, None).astype(np.float32); o[vai] = p
        t += np.clip(np.expm1(mdl.predict(Xte, verbose=0).flatten()), 0, None).astype(np.float32) / 5
        print(f"    F{fi+1} MAE: {mean_absolute_error(y.values[vai], p):.4f}", flush=True)
        del mdl; tf.keras.backend.clear_session(); gc.collect()
    cv_maes['mlp'] = mean_absolute_error(y, o); oof_preds['mlp'] = o; test_preds['mlp'] = t
    save_ckpt(ckpt_path, {'oof': o, 'test': t, 'cv_mae': cv_maes['mlp']}, feature_cols=feature_cols)
print(f"  CV: {cv_maes['mlp']:.4f}", flush=True)

# --- Model 8: TabNet ---
print("\n  [8] TabNet", flush=True)
from pytorch_tabnet.tab_model import TabNetRegressor
ckpt_path = 'output/ckpt_phase21c_tabnet.pkl'; ckpt = load_ckpt(ckpt_path, expected_features=feature_cols)
if ckpt: oof_preds['tabnet'] = ckpt['oof']; test_preds['tabnet'] = ckpt['test']; cv_maes['tabnet'] = ckpt['cv_mae']
else:
    o, t = np.zeros(len(X), dtype=np.float32), np.zeros(len(X_test), dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        import torch; torch.manual_seed(42); np.random.seed(42)
        rs = RobustScaler(); Xtr = np.clip(np.nan_to_num(rs.fit_transform(X.iloc[tri]), nan=0.0, posinf=5.0, neginf=-5.0), -5, 5).astype('float32')
        Xva = np.clip(np.nan_to_num(rs.transform(X.iloc[vai]), nan=0.0, posinf=5.0, neginf=-5.0), -5, 5).astype('float32')
        Xte = np.clip(np.nan_to_num(rs.transform(X_test), nan=0.0, posinf=5.0, neginf=-5.0), -5, 5).astype('float32')
        mdl = TabNetRegressor(n_d=32, n_a=32, n_steps=3, gamma=1.3, lambda_sparse=1e-3, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2), scheduler_params={"step_size": 10, "gamma": 0.9}, scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax', seed=42, verbose=10, device_name='cuda')
        mdl.fit(X_train=Xtr, y_train=y.values[tri].reshape(-1, 1).astype('float32'), eval_set=[(Xva, y.values[vai].reshape(-1, 1).astype('float32'))], eval_metric=['mae'], max_epochs=50, patience=15, batch_size=2048, virtual_batch_size=256, num_workers=0, drop_last=False)
        p = np.clip(mdl.predict(Xva).flatten(), 0, None).astype(np.float32); o[vai] = p
        t += np.clip(mdl.predict(Xte).flatten(), 0, None).astype(np.float32) / 5
        print(f"    F{fi+1} MAE: {mean_absolute_error(y.values[vai], p):.4f}", flush=True)
        del mdl; gc.collect()
    cv_maes['tabnet'] = mean_absolute_error(y, o); oof_preds['tabnet'] = o; test_preds['tabnet'] = t
    save_ckpt(ckpt_path, {'oof': o, 'test': t, 'cv_mae': cv_maes['tabnet']}, feature_cols=feature_cols)
print(f"  CV: {cv_maes['tabnet']:.4f}", flush=True)

# ##############################################################
# Part 5: Ensemble
# ##############################################################
print("\n=== Ensemble ===", flush=True)
model_names = list(oof_preds.keys())
oof_matrix = np.column_stack([oof_preds[n] for n in model_names])
test_matrix = np.column_stack([test_preds[n] for n in model_names])

def objective(w, om, yt):
    w = w / w.sum()
    return np.abs((om * w).sum(axis=1) - yt).mean()

x0 = np.ones(len(model_names)) / len(model_names)
res = minimize(objective, x0, args=(oof_matrix, y.values), method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})
best_w = res.x / res.x.sum()
ensemble_cv = res.fun

print(f"\n  Phase 21C Ensemble CV: {ensemble_cv:.4f}", flush=True)
print(f"  Phase 16 baseline:    8.4403", flush=True)
print(f"  Improvement:          {8.4403 - ensemble_cv:+.4f}", flush=True)
print(f"\n  Weights:", flush=True)
for n, w in zip(model_names, best_w):
    print(f"    {n:12s}: {w:.4f} (CV {cv_maes[n]:.4f})", flush=True)

# ##############################################################
# Part 6: Submission
# ##############################################################
print("\n=== Submission ===", flush=True)
test_pred = np.clip((test_matrix * best_w).sum(axis=1), 0, 500)
sub = sample_sub.copy()
sub['avg_delay_minutes_next_30m'] = test_pred.astype('float32')
sub.to_csv('output/submission_phase21c.csv', index=False)
shutil.copy('output/submission_phase21c.csv', os.path.join(DRIVE_CKPT_DIR, 'submission_phase21c.csv'))
assert (sub['ID'] == sample_sub['ID']).all()
assert (sub['avg_delay_minutes_next_30m'] >= 0).all()
print(f"  submission_phase21c.csv saved", flush=True)
print(sub['avg_delay_minutes_next_30m'].describe(), flush=True)

# ##############################################################
# Part 7: Results
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Phase 21C Results (MLP RMSE+MAE loss isolated test) ===", flush=True)
print("=" * 60, flush=True)
print(f"Change: MLP loss = (RMSE+MAE)/2 (was: mae)", flush=True)
print(f"Total features: {len(final_feature_cols)}", flush=True)
for n in model_names:
    print(f"  {n:12s}: {cv_maes[n]:.4f}", flush=True)
print(f"\nPhase 21C Ensemble CV: {ensemble_cv:.4f}", flush=True)
print(f"Phase 16 baseline:     8.4403", flush=True)
print(f"Phase 16 MLP CV:       8.5887", flush=True)
print(f"Delta (ensemble):      {8.4403 - ensemble_cv:+.4f}", flush=True)
print(f"Delta (MLP):           {8.5887 - cv_maes.get('mlp', 0):+.4f}", flush=True)
if cv_maes.get('mlp', 99) < 8.5887:
    print(f"  → MLP RMSE+MAE loss is EFFECTIVE (MLP improved)", flush=True)
else:
    print(f"  → MLP RMSE+MAE loss has NO EFFECT on MLP", flush=True)

if os.path.exists('output/phase13s2_analysis/layout_mae_ranking.csv'):
    hdf = pd.read_csv('output/phase13s2_analysis/layout_mae_ranking.csv')
    if 'difficulty' in hdf.columns:
        hids = hdf[hdf['difficulty'] == 'hard']['layout_id'].tolist()
        hmask = train_fe['layout_id'].isin(hids).values
        final_oof = (oof_matrix * best_w).sum(axis=1)
        print(f"\nHard layout MAE: {np.abs(final_oof[hmask] - y.values[hmask]).mean():.4f} (baseline 18.784)", flush=True)

# Feature importance
if 'm1_models' in dir() and len(m1_models) > 0:
    imp = pd.DataFrame({'feature': feature_cols, 'importance': m1_models[0].feature_importances_}).sort_values('importance', ascending=False)
    top30 = imp.head(30).sort_values('importance')
    n_new = top30['feature'].isin(selected_2nd).sum()
    print(f"\n  New 2nd-order in Top 30: {n_new}", flush=True)
    colors = ['crimson' if f in selected_2nd else ('orangered' if f.startswith(('s_', 'l_', 'h_', 'lh_', 'r_')) else 'steelblue') for f in top30['feature']]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top30['feature'], top30['importance'], color=colors)
    ax.set_title('Feature Importance Top 30 (Phase 21C)')
    ax.legend(handles=[Patch(facecolor='steelblue', label='Base'), Patch(facecolor='orangered', label='P15 Agg'), Patch(facecolor='crimson', label='2nd-order')], loc='lower right')
    plt.tight_layout(); plt.savefig('output/feature_importance_phase21c.png', dpi=150, bbox_inches='tight')
    print("  feature_importance_phase21c.png saved", flush=True)

print("\n=== Phase 21C Complete ===", flush=True)
