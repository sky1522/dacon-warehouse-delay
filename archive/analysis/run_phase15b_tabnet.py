"""
Phase 15B: TabNet retrained + 8-model ensemble + P13s1 reblend
- Reproduce Phase 15 features exactly (Part 1-3 from run_phase15_fe.py)
- Train only TabNet (7 other models loaded from checkpoints)
- 8-model Nelder-Mead ensemble
- Blend with Phase 13s1
"""

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
import torch
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

DRIVE_CKPT_DIR = '/content/drive/MyDrive/dacon_ckpt'
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)


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
# Part 1: Phase 15 Base Features (exact copy from run_phase15_fe.py)
# ============================================================
print("=" * 60, flush=True)
print("=== Part 1: Phase 15 Base Features ===", flush=True)
print("=" * 60, flush=True)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
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

# Phase 3B Interactions
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

# Onset Features (8)
print("=== Onset Features ===", flush=True)
def compute_onset_idx(group, flag_col, timeslot_col='implicit_timeslot'):
    flags = group[flag_col].values
    slots = group[timeslot_col].values
    result = np.full(len(group), -1, dtype=np.float32)
    first_idx = -1
    for i in range(len(flags)):
        if i == 0: continue
        if flags[i-1] > 0 and first_idx == -1: first_idx = slots[i-1]
        result[i] = first_idx
    return pd.Series(result, index=group.index)

combined['_charging_flag'] = (combined['robot_charging'] > 0).astype(np.float32)
combined['_charging_flag_prev'] = combined.groupby('scenario_id')['_charging_flag'].shift(1).fillna(0)
combined['charging_ever_started'] = combined.groupby('scenario_id')['_charging_flag_prev'].cummax()
combined['charging_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(lambda g: compute_onset_idx(g, '_charging_flag'))
combined['charging_steps_since_start'] = np.where(combined['charging_start_idx'] >= 0, combined['implicit_timeslot'] - combined['charging_start_idx'], -1).astype(np.float32)
combined['charging_started_early'] = ((combined['charging_start_idx'] >= 0) & (combined['charging_start_idx'] < 5)).astype(np.float32)
combined['_queue_flag'] = (combined['charge_queue_length'] > 0).astype(np.float32)
combined['_queue_flag_prev'] = combined.groupby('scenario_id')['_queue_flag'].shift(1).fillna(0)
combined['queue_ever_started'] = combined.groupby('scenario_id')['_queue_flag_prev'].cummax()
combined['queue_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(lambda g: compute_onset_idx(g, '_queue_flag'))
combined['_congestion_flag'] = (combined['congestion_score'] > 0).astype(np.float32)
combined['_congestion_flag_prev'] = combined.groupby('scenario_id')['_congestion_flag'].shift(1).fillna(0)
combined['congestion_ever_started'] = combined.groupby('scenario_id')['_congestion_flag_prev'].cummax()
combined['congestion_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(lambda g: compute_onset_idx(g, '_congestion_flag'))
combined.drop(columns=['_charging_flag', '_charging_flag_prev', '_queue_flag', '_queue_flag_prev', '_congestion_flag', '_congestion_flag_prev'], inplace=True)
print("  Onset 8 done", flush=True)

# Expanding Mean (30)
print("=== Expanding Mean ===", flush=True)
expanding_cols_mean = ['order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
                        'heavy_item_ratio', 'robot_active', 'battery_mean', 'low_battery_ratio',
                        'congestion_score', 'max_zone_density', 'pack_utilization', 'loading_dock_util',
                        'charge_queue_length', 'fault_count_15m', 'avg_trip_distance']
for i, col in enumerate(expanding_cols_mean):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    expmean = shifted.groupby(combined['scenario_id']).expanding().mean().droplevel(0).sort_index()
    combined[f'{col}_expmean_prev'] = expmean.astype(np.float32)
    combined[f'{col}_delta_expmean'] = (combined[col] - combined[f'{col}_expmean_prev']).astype(np.float32)
    if (i + 1) % 5 == 0: print(f"  Expanding Mean: {i+1}/{len(expanding_cols_mean)}", flush=True)

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

# Competitor Features (54)
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
dynamic_cols = ['order_inflow_15m', 'battery_mean', 'battery_std', 'low_battery_ratio', 'robot_active', 'robot_idle', 'robot_charging', 'congestion_score', 'max_zone_density', 'pack_utilization', 'loading_dock_util', 'charge_queue_length', 'fault_count_15m', 'avg_trip_distance', 'unique_sku_15m']
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

# Expanding Extension B (20)
print("=== Expanding Extension ===", flush=True)
expanding_ext_cols = ['order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization', 'loading_dock_util', 'robot_active', 'low_battery_ratio', 'avg_trip_distance', 'unique_sku_15m', 'max_zone_density']
for i, col in enumerate(expanding_ext_cols):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    grp = shifted.groupby(combined['scenario_id'])
    combined[f'{col}_expstd_prev'] = grp.expanding().std().droplevel(0).sort_index().astype(np.float32)
    combined[f'{col}_expmax_prev'] = grp.expanding().max().droplevel(0).sort_index().astype(np.float32)
    if (i + 1) % 5 == 0: print(f"  {i+1}/{len(expanding_ext_cols)}", flush=True)

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
arrival_rate = (combined['order_inflow_15m'] / 15.0).astype(np.float32)
combined['q_arrival_rate'] = arrival_rate
combined['q_expected_charge_wait'] = (combined['charge_queue_length'] / (arrival_rate + EPS)).astype(np.float32)
combined['q_effective_service_robot'] = (combined['robot_active'] / (combined['avg_trip_distance'] + EPS)).astype(np.float32)
combined['q_arrival_service_gap'] = (arrival_rate - combined['q_effective_service_robot']).astype(np.float32)
combined['q_throughput_robot'] = (combined['robot_active'] * (1.0 - combined['congestion_score'])).astype(np.float32)
combined['q_throughput_pack'] = (combined['pack_station_count'] * combined['pack_utilization']).astype(np.float32)
stages_df = combined[['q_rho_robot', 'q_rho_charger', 'q_rho_pack', 'q_rho_loading']]
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

# EDA Features (7)
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
LOW_CORR = ['layout_compactness', 'zone_dispersion', 'one_way_ratio', 'aisle_width_avg', 'floor_area_sqm', 'ceiling_height_m', 'building_age_years', 'intersection_count', 'fire_sprinkler_count', 'emergency_exit_count']
DERIVED = ['robot_per_area', 'congestion_per_width', 'zone_density_per_width', 'order_per_sqm', 'warehouse_volume', 'intersection_density', 'pack_station_density', 'charger_density', 'robot_density_layout', 'movement_friction', 'layout_compact_x_dispersion', 'one_way_friction']
combined = combined.drop(columns=[c for c in LOW_CORR + DERIVED if c in combined.columns])

drop_cols_meta = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']
base_feature_cols = [c for c in combined.columns if c not in drop_cols_meta and combined[c].dtype in [np.float32, np.float64, np.int32, np.int64, np.int8]]
print(f"  Base features: {len(base_feature_cols)}", flush=True)

# ============================================================
# Part 2: Large-scale Aggregation (exact copy from Phase 15)
# ============================================================
print("\n=== Aggregation Features ===", flush=True)

SCENARIO_AGG_COLS = ['congestion_score', 'max_zone_density', 'robot_charging', 'low_battery_ratio', 'robot_idle', 'order_inflow_15m', 'battery_mean', 'charge_queue_length', 'avg_charge_wait', 'near_collision_15m', 'blocked_path_15m', 'unique_sku_15m', 'fault_count_15m', 'avg_recovery_time', 'sku_concentration', 'urgent_order_ratio', 'battery_std', 'robot_utilization', 'robot_active', 'pack_utilization', 'loading_dock_util', 'heavy_item_ratio', 'manual_override_ratio', 'cold_chain_ratio', 'outbound_truck_wait_min', 'packaging_material_cost']
SCENARIO_AGG_COLS = [c for c in SCENARIO_AGG_COLS if c in combined.columns]
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
for col in SCENARIO_AGG_COLS[:15]:
    combined[f's_{col}_dist_from_max'] = (combined[f's_{col}_max'] - combined[col]).astype('float32')
    combined[f's_{col}_rank_in_scenario'] = combined.groupby('scenario_id')[col].rank(pct=True).astype('float32')
print(f"  Scenario feats: {len([c for c in combined.columns if c.startswith('s_')])}", flush=True)

LAYOUT_AGG_COLS = [c for c in ['congestion_score', 'robot_active', 'robot_charging', 'order_inflow_15m', 'pack_utilization', 'charge_queue_length', 'urgent_order_ratio', 'max_zone_density', 'battery_mean', 'robot_utilization', 'sku_concentration'] if c in combined.columns]
for col in LAYOUT_AGG_COLS:
    combined[f'l_{col}_mean'] = combined.groupby('layout_id')[col].transform('mean').astype('float32')
    combined[f'l_{col}_std'] = combined.groupby('layout_id')[col].transform('std').astype('float32')
    combined[f'l_{col}_max'] = combined.groupby('layout_id')[col].transform('max').astype('float32')
    combined[f'l_{col}_p90'] = combined.groupby('layout_id')[col].transform(lambda x: x.quantile(0.9)).astype('float32')
    combined[f'l_{col}_ratio'] = (combined[col] / (combined[f'l_{col}_mean'] + 1e-6)).astype('float32')
print(f"  Layout feats: {len([c for c in combined.columns if c.startswith('l_')])}", flush=True)

HOUR_AGG_COLS = [c for c in ['congestion_score', 'robot_active', 'order_inflow_15m', 'pack_utilization', 'charge_queue_length', 'urgent_order_ratio'] if c in combined.columns]
combined['shift_hour_filled'] = combined['shift_hour'].fillna(-1)
for col in HOUR_AGG_COLS:
    combined[f'h_{col}_mean'] = combined.groupby('shift_hour_filled')[col].transform('mean').astype('float32')
    combined[f'h_{col}_std'] = combined.groupby('shift_hour_filled')[col].transform('std').astype('float32')
    combined[f'h_{col}_deviation'] = (combined[col] - combined[f'h_{col}_mean']).astype('float32')
combined = combined.drop(columns=['shift_hour_filled'])

CROSS_AGG_COLS = [c for c in ['congestion_score', 'robot_active', 'order_inflow_15m', 'pack_utilization', 'charge_queue_length'] if c in combined.columns]
combined['shift_hour_filled'] = combined['shift_hour'].fillna(-1)
combined['lh_key'] = combined['layout_id'].astype(str) + '_' + combined['shift_hour_filled'].astype(str)
for col in CROSS_AGG_COLS:
    combined[f'lh_{col}_mean'] = combined.groupby('lh_key')[col].transform('mean').astype('float32')
    combined[f'lh_{col}_std'] = combined.groupby('lh_key')[col].transform('std').astype('float32')
combined = combined.drop(columns=['shift_hour_filled', 'lh_key'])

combined['r_pack_per_robot'] = (combined['pack_station_count'] / (combined['robot_active'] + 1.0)).astype('float32')
combined['r_charger_per_robot'] = (combined['charger_count'] / (combined['robot_active'] + 1.0)).astype('float32')
combined['r_pack_util_x_inflow'] = (combined['pack_utilization'] * combined['order_inflow_15m'] / 100.0).astype('float32')
combined['r_congestion_x_robot_ratio'] = (combined['congestion_score'] * combined['robot_active'] / (combined['robot_total'] + 1.0)).astype('float32')
combined['r_queue_per_charger'] = (combined['charge_queue_length'] / (combined['charger_count'] + 1.0)).astype('float32')
combined['r_active_per_sqm'] = (combined['robot_active'] / 2.0).astype('float32')  # floor_area_sqm dropped

new_feat_cols = [c for c in combined.columns if c.startswith(('s_', 'l_', 'h_', 'lh_', 'r_'))]
for col in new_feat_cols:
    combined[col] = np.nan_to_num(combined[col].astype('float32').values, nan=0.0, posinf=1e6, neginf=-1e6)
print(f"  Total new features: {len(new_feat_cols)}", flush=True)

# ============================================================
# Feature Selection (identical to Phase 15)
# ============================================================
print("\n=== Feature Selection ===", flush=True)
float64_cols = combined.select_dtypes(include='float64').columns
combined[float64_cols] = combined[float64_cols].astype(np.float32)
combined = combined.sort_values('_original_idx').reset_index(drop=True)

y_train = combined.loc[combined['_is_train'] == 1, 'avg_delay_minutes_next_30m'].values
layout_ids_train = combined.loc[combined['_is_train'] == 1, 'layout_id'].values
temp_feature_cols = base_feature_cols + [c for c in new_feat_cols if c in combined.columns]
X_train_temp = combined.loc[combined['_is_train'] == 1, temp_feature_cols].fillna(0).astype('float32')

gkf = GroupKFold(n_splits=5)
tr_idx, va_idx = next(gkf.split(X_train_temp, y_train, groups=layout_ids_train))
lgb_selector = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, learning_rate=0.05,
                                   num_leaves=127, max_depth=8, min_child_samples=50,
                                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
                                   verbosity=-1, random_state=42)
lgb_selector.fit(X_train_temp.iloc[tr_idx], y_train[tr_idx],
                 eval_set=[(X_train_temp.iloc[va_idx], y_train[va_idx])],
                 callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

imp_df = pd.DataFrame({'feature': temp_feature_cols, 'importance': lgb_selector.feature_importances_}).sort_values('importance', ascending=False)
new_imp = imp_df[imp_df['feature'].isin(new_feat_cols)]
selected_new_feats = new_imp.head(200)['feature'].tolist()
dropped_new_feats = [f for f in new_feat_cols if f not in selected_new_feats and f in combined.columns]
final_feature_cols = base_feature_cols + selected_new_feats
print(f"  Final features: {len(final_feature_cols)}", flush=True)
combined = combined.drop(columns=[c for c in dropped_new_feats if c in combined.columns])
del X_train_temp, lgb_selector; gc.collect()

# ============================================================
# Split + CV
# ============================================================
train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()
del combined; gc.collect()

feature_cols = final_feature_cols
X = train_fe[feature_cols]
y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y)
X_test = test_fe[feature_cols]
y_train_arr = y.values

assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID order mismatch!"

y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(sgkf.split(X, y_binned, groups=train_fe['layout_id']))
print("  CV validated", flush=True)

# ============================================================
# Part 2: TabNet Training
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== TabNet Training ===", flush=True)
print("=" * 60, flush=True)

from pytorch_tabnet.tab_model import TabNetRegressor

ckpt_path = 'output/ckpt_phase15_tabnet.pkl'
ckpt = load_ckpt(ckpt_path)
if ckpt is not None:
    tabnet_oof = ckpt['oof']
    tabnet_test = ckpt['test']
    tabnet_cv = ckpt['cv_mae']
    print(f"  Cache hit (CV MAE: {tabnet_cv:.4f})", flush=True)
else:
    tabnet_oof = np.zeros(len(X), dtype='float32')
    tabnet_test = np.zeros(len(X_test), dtype='float32')

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n  TabNet Fold {fold_idx+1}/5...", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)

        scaler = RobustScaler()
        X_tr_scaled = np.clip(scaler.fit_transform(X.iloc[tr_idx]), -5, 5).astype('float32')
        X_va_scaled = np.clip(scaler.transform(X.iloc[va_idx]), -5, 5).astype('float32')
        X_te_scaled = np.clip(scaler.transform(X_test), -5, 5).astype('float32')

        X_tr_scaled = np.nan_to_num(X_tr_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
        X_va_scaled = np.nan_to_num(X_va_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
        X_te_scaled = np.nan_to_num(X_te_scaled, nan=0.0, posinf=5.0, neginf=-5.0)

        y_tr = y_train_arr[tr_idx].reshape(-1, 1).astype('float32')
        y_va = y_train_arr[va_idx].reshape(-1, 1).astype('float32')

        model = TabNetRegressor(
            n_d=32, n_a=32, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax', seed=42, verbose=10, device_name='cuda',
        )
        model.fit(
            X_train=X_tr_scaled, y_train=y_tr,
            eval_set=[(X_va_scaled, y_va)],
            eval_metric=['mae'],
            max_epochs=50, patience=15,
            batch_size=2048, virtual_batch_size=256,
            num_workers=0, drop_last=False,
        )

        pred_va = np.clip(model.predict(X_va_scaled).flatten(), 0, None).astype('float32')
        tabnet_oof[va_idx] = pred_va
        tabnet_test += np.clip(model.predict(X_te_scaled).flatten(), 0, None).astype('float32') / 5

        fold_mae = np.abs(pred_va - y_train_arr[va_idx]).mean()
        print(f"    Fold {fold_idx+1} MAE: {fold_mae:.4f}", flush=True)
        del model; gc.collect()

    tabnet_cv = np.abs(tabnet_oof - y_train_arr).mean()
    print(f"\n  TabNet CV MAE: {tabnet_cv:.4f}", flush=True)

    ckpt_data = {'oof': tabnet_oof, 'test': tabnet_test, 'cv_mae': tabnet_cv}
    with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt_data, f)
    shutil.copy(ckpt_path, os.path.join(DRIVE_CKPT_DIR, 'ckpt_phase15_tabnet.pkl'))
    print(f"  Saved: {ckpt_path}", flush=True)

# ============================================================
# Part 3: 8-model Ensemble
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== 8-model Ensemble ===", flush=True)
print("=" * 60, flush=True)

model_ckpt_names = {
    'lgb_raw': 'ckpt_phase15_lgb_raw.pkl',
    'lgb_huber': 'ckpt_phase15_lgb_huber.pkl',
    'lgb_sqrt': 'ckpt_phase15_lgb_sqrt.pkl',
    'xgb': 'ckpt_phase15_xgb.pkl',
    'cat_log1p': 'ckpt_phase15_cat_log1p.pkl',
    'cat_raw': 'ckpt_phase15_cat_raw.pkl',
    'mlp': 'ckpt_phase15_mlp.pkl',
    'tabnet': 'ckpt_phase15_tabnet.pkl',
}

oofs = {}
tests = {}
for name, ckpt_file in model_ckpt_names.items():
    ckpt = load_ckpt(f'output/{ckpt_file}')
    if ckpt is None:
        print(f"  WARNING: {ckpt_file} not found!", flush=True)
        continue
    oofs[name] = ckpt['oof']
    tests[name] = ckpt['test']
    cv_val = np.abs(ckpt['oof'] - y_train_arr).mean()
    print(f"  {name}: CV {cv_val:.4f}", flush=True)

model_names = list(oofs.keys())
oof_matrix = np.column_stack([oofs[n] for n in model_names])
test_matrix = np.column_stack([tests[n] for n in model_names])

# Nelder-Mead
def objective(w, oof_mat, y_true):
    w = w / w.sum()
    return np.abs((oof_mat * w).sum(axis=1) - y_true).mean()

x0 = np.ones(len(model_names)) / len(model_names)
result = minimize(objective, x0, args=(oof_matrix, y_train_arr),
                  method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})
best_weights = result.x / result.x.sum()
ensemble_cv_8 = result.fun

print(f"\n=== Phase 15 Full (8 models) ===", flush=True)
print(f"CV MAE: {ensemble_cv_8:.4f}", flush=True)
print(f"Phase 15 (7 models): 8.4553", flush=True)
print(f"Phase 13s1 baseline: 8.5668", flush=True)
print(f"\nWeights:", flush=True)
for n, w in zip(model_names, best_weights):
    print(f"  {n:12s}: {w:.4f}", flush=True)

# Test prediction
test_pred_8 = np.clip((test_matrix * best_weights).sum(axis=1), 0, 500)

# Submission (Phase 15 full)
sub = sample_sub.copy()
sub['avg_delay_minutes_next_30m'] = test_pred_8.astype('float32')
sub.to_csv('output/submission_phase15_full.csv', index=False)
shutil.copy('output/submission_phase15_full.csv', os.path.join(DRIVE_CKPT_DIR, 'submission_phase15_full.csv'))
assert (sub['ID'] == sample_sub['ID']).all()
assert (sub['avg_delay_minutes_next_30m'] >= 0).all()
print(f"  submission_phase15_full.csv saved", flush=True)

# ============================================================
# Part 4: Blend with Phase 13s1
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Blend with Phase 13s1 ===", flush=True)
print("=" * 60, flush=True)

p13_models = ['lgb_raw', 'lgb_huber', 'lgb_sqrt', 'xgb', 'cat_log1p', 'cat_raw', 'mlp', 'tabnet']
p13_weights = {'lgb_raw': 0.0752, 'lgb_huber': 0.2796, 'lgb_sqrt': -0.0259,
               'xgb': 0.0728, 'cat_log1p': 0.1901, 'cat_raw': -0.1517,
               'mlp': 0.3483, 'tabnet': 0.2116}

p13_oof = np.zeros(len(y_train_arr), dtype='float32')
p13_test = np.zeros(len(test_pred_8), dtype='float32')
p13_loaded = True

for name in p13_models:
    ckpt = load_ckpt(f'output/ckpt_phase13s1_{name}.pkl')
    if ckpt is None:
        print(f"  WARNING: Phase 13s1 {name} not found, skipping blend", flush=True)
        p13_loaded = False
        break
    p13_oof += p13_weights[name] * ckpt['oof']
    p13_test += p13_weights[name] * ckpt['test']

if p13_loaded:
    p15_full_oof = (oof_matrix * best_weights).sum(axis=1)
    p15_full_test = (test_matrix * best_weights).sum(axis=1)

    # Grid search blend weight
    best_w, best_mae = 0, 999
    for w in np.arange(0, 1.01, 0.01):
        blend = (1 - w) * p15_full_oof + w * p13_oof
        mae = np.abs(blend - y_train_arr).mean()
        if mae < best_mae:
            best_mae = mae
            best_w = w

    print(f"Best: P15 weight={1-best_w:.2f}, P13s1 weight={best_w:.2f}", flush=True)
    print(f"Blend CV: {best_mae:.4f}", flush=True)
    print(f"Phase 15 full CV: {ensemble_cv_8:.4f}", flush=True)
    print(f"Improvement: {ensemble_cv_8 - best_mae:+.4f}", flush=True)

    blend_test = np.clip((1 - best_w) * p15_full_test + best_w * p13_test, 0, 500)
    sub_blend = sample_sub.copy()
    sub_blend['avg_delay_minutes_next_30m'] = blend_test.astype('float32')
    sub_blend.to_csv('output/submission_phase15b_blend.csv', index=False)
    shutil.copy('output/submission_phase15b_blend.csv', os.path.join(DRIVE_CKPT_DIR, 'submission_phase15b_blend.csv'))
    print(f"  submission_phase15b_blend.csv saved", flush=True)
else:
    print("  Phase 13s1 checkpoints not available, skipping blend", flush=True)

# ============================================================
# Results
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 15B Results ===", flush=True)
print("=" * 60, flush=True)
print(f"TabNet CV MAE: {tabnet_cv:.4f}", flush=True)
print(f"Phase 15 Full (8 models) CV: {ensemble_cv_8:.4f}", flush=True)
print(f"Phase 15 (7 models) baseline: 8.4553", flush=True)
if p13_loaded:
    print(f"Blend CV: {best_mae:.4f}", flush=True)
    print(f"Previous best blend (P15 7m + P13s1): 9.8891 Public", flush=True)
print("\n=== Phase 15B Complete ===", flush=True)
