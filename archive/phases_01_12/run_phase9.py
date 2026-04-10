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
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from matplotlib.patches import Patch
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1. 데이터 준비 (Phase 8 로직 그대로 재사용 — 319개 피처)
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
print("  시계열 피처 64개 생성 완료", flush=True)

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
print("=== Phase 3B 인터랙션 피처 ===", flush=True)
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
print("=== Onset 피처 생성 ===", flush=True)

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
print("  충전 onset...", flush=True)
combined['charging_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_charging_flag'))
combined['charging_steps_since_start'] = np.where(
    combined['charging_start_idx'] >= 0,
    combined['implicit_timeslot'] - combined['charging_start_idx'], -1).astype(np.float32)
combined['charging_started_early'] = (
    (combined['charging_start_idx'] >= 0) & (combined['charging_start_idx'] < 5)).astype(np.float32)

print("  대기열 onset...", flush=True)
combined['_queue_flag'] = (combined['charge_queue_length'] > 0).astype(np.float32)
combined['_queue_flag_prev'] = combined.groupby('scenario_id')['_queue_flag'].shift(1).fillna(0)
combined['queue_ever_started'] = combined.groupby('scenario_id')['_queue_flag_prev'].cummax()
combined['queue_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_queue_flag'))

print("  혼잡 onset...", flush=True)
combined['_congestion_flag'] = (combined['congestion_score'] > 0).astype(np.float32)
combined['_congestion_flag_prev'] = combined.groupby('scenario_id')['_congestion_flag'].shift(1).fillna(0)
combined['congestion_ever_started'] = combined.groupby('scenario_id')['_congestion_flag_prev'].cummax()
combined['congestion_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_congestion_flag'))

combined.drop(columns=['_charging_flag', '_charging_flag_prev', '_queue_flag', '_queue_flag_prev',
                        '_congestion_flag', '_congestion_flag_prev'], inplace=True)
print("  Onset 피처 8개 완료", flush=True)

# --- Expanding Mean (30개) ---
print("=== Expanding Mean 피처 ===", flush=True)
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
print("=== 비선형 피처 ===", flush=True)
combined['battery_mean_below_44'] = np.maximum(44.0 - combined['battery_mean'], 0).astype(np.float32)
combined['low_battery_ratio_above_02'] = np.maximum(combined['low_battery_ratio'] - 0.2, 0).astype(np.float32)
combined['pack_utilization_sq'] = (combined['pack_utilization'] ** 2).astype(np.float32)
combined['loading_dock_util_sq'] = (combined['loading_dock_util'] ** 2).astype(np.float32)
combined['congestion_score_sq'] = (combined['congestion_score'] ** 2).astype(np.float32)
combined['charge_pressure_nl'] = ((combined['robot_charging'] + combined['charge_queue_length']) / (combined['charger_count'] + 1)).astype(np.float32)
combined['charge_pressure_nl_sq'] = (combined['charge_pressure_nl'] ** 2).astype(np.float32)

# --- 위상 (6개) ---
print("=== 위상 피처 ===", flush=True)
combined['is_early_phase'] = (combined['implicit_timeslot'] <= 5).astype(np.float32)
combined['is_mid_phase'] = ((combined['implicit_timeslot'] >= 6) & (combined['implicit_timeslot'] <= 15)).astype(np.float32)
combined['is_late_phase'] = (combined['implicit_timeslot'] >= 16).astype(np.float32)
combined['time_frac'] = (combined['implicit_timeslot'] / 24.0).astype(np.float32)
combined['time_remaining'] = (24 - combined['implicit_timeslot']).astype(np.float32)
combined['time_frac_sq'] = (combined['time_frac'] ** 2).astype(np.float32)

# --- 경쟁자 피처 A (54개) ---
print("=== 경쟁자 피처 ===", flush=True)
# A-1) 로봇 상태 분해 (6개)
combined['robot_total_state'] = combined['robot_active'] + combined['robot_idle'] + combined['robot_charging']
combined['robot_total_gap'] = combined['robot_total_state'] - combined['robot_total']
combined['robot_active_share'] = combined['robot_active'] / (combined['robot_total_state'] + 1)
combined['robot_idle_share'] = combined['robot_idle'] / (combined['robot_total_state'] + 1)
combined['robot_charging_share'] = combined['robot_charging'] / (combined['robot_total_state'] + 1)
combined['charging_to_active_ratio'] = combined['robot_charging'] / (combined['robot_active'] + 1)

# A-2) 수요-용량 비율 (15개)
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

# A-3) 복합 인터랙션 (12개)
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

# A-4) layout 밀도 (8개)
combined['warehouse_volume'] = combined['floor_area_sqm'] * combined['ceiling_height_m']
combined['intersection_density'] = combined['intersection_count'] / (combined['floor_area_sqm'] + 1)
combined['pack_station_density'] = combined['pack_station_count'] / (combined['floor_area_sqm'] + 1)
combined['charger_density'] = combined['charger_count'] / (combined['floor_area_sqm'] + 1)
combined['robot_density_layout'] = combined['robot_total'] / (combined['floor_area_sqm'] + 1)
combined['movement_friction'] = combined['intersection_count'] / (combined['aisle_width_avg'] + 0.01)
combined['layout_compact_x_dispersion'] = combined['layout_compactness'] * combined['zone_dispersion']
combined['one_way_friction'] = combined['one_way_ratio'] * combined['intersection_count'] / (combined['aisle_width_avg'] + 0.01)

# A-5) missing indicator (3개)
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

# A-6) rolling max + 편차 (10개)
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
print("\n=== float32 변환 + 데이터 분리 ===", flush=True)
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
# 2. Optuna 재튜닝 (319개 피처 기준, 30 trials)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Optuna 재튜닝 (30 trials) ===", flush=True)
print("=" * 60, flush=True)

def optuna_objective(trial):
    params = {
        'objective': 'mae',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    oof = np.zeros(len(X))
    for tr_idx, va_idx in folds:
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sample_w[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va_idx] = np.clip(m.predict(X.iloc[va_idx]), 0, None)
    return mean_absolute_error(y, oof)

def optuna_callback(study, trial):
    if (trial.number + 1) % 10 == 0:
        print(f"  Trial {trial.number+1}/30: best MAE={study.best_value:.4f}", flush=True)

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optuna_objective, n_trials=30, callbacks=[optuna_callback])

optuna_best_params = study.best_params
optuna_best_mae = study.best_value
print(f"\n  Optuna 최적 MAE: {optuna_best_mae:.4f} (기존 Phase3B params 기준 8.7494)", flush=True)
print(f"  Optuna best params: {optuna_best_params}", flush=True)

# ============================================================
# 3. Multi-Seed 6모델 학습
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Multi-Seed 6모델 학습 ===", flush=True)
print("=" * 60, flush=True)

SEEDS = [42, 123, 777]

def train_lgb_multiseed(X, y_target, X_test, folds, params_base, sample_w, inv_fn, model_name):
    """3-seed 평균 OOF/test 반환"""
    all_oof = []
    all_test = []
    for seed in SEEDS:
        params = params_base.copy()
        params['random_state'] = seed
        oof_s = np.zeros(len(X), dtype=np.float32)
        test_s = np.zeros(len(X_test), dtype=np.float32)
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**params)
            m.fit(X.iloc[tr_idx], y_target.iloc[tr_idx], sample_weight=sample_w[tr_idx],
                  eval_set=[(X.iloc[va_idx], y_target.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof_s[va_idx] = inv_fn(m.predict(X.iloc[va_idx]))
            test_s += inv_fn(m.predict(X_test)) / 5
        all_oof.append(oof_s)
        all_test.append(test_s)
        mae_s = mean_absolute_error(y, oof_s)
        print(f"    {model_name} seed={seed}: CV MAE {mae_s:.4f}", flush=True)
    avg_oof = np.mean(all_oof, axis=0).astype(np.float32)
    avg_test = np.mean(all_test, axis=0).astype(np.float32)
    avg_mae = mean_absolute_error(y, avg_oof)
    print(f"    {model_name} seed-avg: CV MAE {avg_mae:.4f}", flush=True)
    return avg_oof, avg_test, avg_mae

def train_xgb_multiseed(X, y_target, X_test, folds, params_base, sample_w, inv_fn, model_name):
    all_oof = []
    all_test = []
    for seed in SEEDS:
        oof_s = np.zeros(len(X), dtype=np.float32)
        test_s = np.zeros(len(X_test), dtype=np.float32)
        for tr_idx, va_idx in folds:
            m = xgb.XGBRegressor(**{**params_base, 'random_state': seed})
            m.fit(X.iloc[tr_idx], y_target.iloc[tr_idx], sample_weight=sample_w[tr_idx],
                  eval_set=[(X.iloc[va_idx], y_target.iloc[va_idx])], verbose=False)
            oof_s[va_idx] = inv_fn(m.predict(X.iloc[va_idx]))
            test_s += inv_fn(m.predict(X_test)) / 5
        all_oof.append(oof_s)
        all_test.append(test_s)
        mae_s = mean_absolute_error(y, oof_s)
        print(f"    {model_name} seed={seed}: CV MAE {mae_s:.4f}", flush=True)
    avg_oof = np.mean(all_oof, axis=0).astype(np.float32)
    avg_test = np.mean(all_test, axis=0).astype(np.float32)
    avg_mae = mean_absolute_error(y, avg_oof)
    print(f"    {model_name} seed-avg: CV MAE {avg_mae:.4f}", flush=True)
    return avg_oof, avg_test, avg_mae

def train_cat_multiseed(X, y_target, X_test, folds, cat_params_base, sample_w, inv_fn, model_name):
    all_oof = []
    all_test = []
    for seed in SEEDS:
        oof_s = np.zeros(len(X), dtype=np.float32)
        test_s = np.zeros(len(X_test), dtype=np.float32)
        for tr_idx, va_idx in folds:
            params = {**cat_params_base, 'random_seed': seed}
            train_pool = Pool(X.iloc[tr_idx], y_target.iloc[tr_idx], weight=sample_w[tr_idx])
            eval_pool = Pool(X.iloc[va_idx], y_target.iloc[va_idx])
            m = CatBoostRegressor(**params)
            m.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=100)
            oof_s[va_idx] = inv_fn(m.predict(X.iloc[va_idx]))
            test_s += inv_fn(m.predict(X_test)) / 5
        all_oof.append(oof_s)
        all_test.append(test_s)
        mae_s = mean_absolute_error(y, oof_s)
        print(f"    {model_name} seed={seed}: CV MAE {mae_s:.4f}", flush=True)
    avg_oof = np.mean(all_oof, axis=0).astype(np.float32)
    avg_test = np.mean(all_test, axis=0).astype(np.float32)
    avg_mae = mean_absolute_error(y, avg_oof)
    print(f"    {model_name} seed-avg: CV MAE {avg_mae:.4f}", flush=True)
    return avg_oof, avg_test, avg_mae

clip0 = lambda x: np.clip(x, 0, None).astype(np.float32)
clip0_expm1 = lambda x: np.clip(np.expm1(x), 0, None).astype(np.float32)
clip0_sq = lambda x: np.clip(x ** 2, 0, None).astype(np.float32)

oof_preds = {}
test_preds = {}
cv_maes = {}

# 모델 1: LightGBM raw+MAE (Optuna NEW params)
print("\n  [모델 1] LightGBM raw+MAE (Optuna NEW params)...", flush=True)
m1_params = {
    'objective': 'mae', 'n_estimators': 2000,
    'n_jobs': -1, 'verbose': -1,
    **optuna_best_params,
}
m1_oof, m1_test, m1_cv = train_lgb_multiseed(X, y, X_test, folds, m1_params, sample_w, clip0, 'LGB_raw_mae')
oof_preds['lgb_raw_mae'] = m1_oof
test_preds['lgb_raw_mae'] = m1_test
cv_maes['lgb_raw_mae'] = m1_cv

# 모델 1의 단일 seed 모델 저장 (피처 중요도용)
m1_imp_model = lgb.LGBMRegressor(**{**m1_params, 'random_state': 42})
m1_imp_model.fit(X.iloc[folds[0][0]], y.iloc[folds[0][0]], sample_weight=sample_w[folds[0][0]],
                 eval_set=[(X.iloc[folds[0][1]], y.iloc[folds[0][1]])],
                 callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

# 모델 2: LightGBM log1p+Huber
print("\n  [모델 2] LightGBM log1p+Huber...", flush=True)
m2_params = {
    'objective': 'huber', 'huber_delta': 0.9, 'n_estimators': 2000, 'learning_rate': 0.03,
    'num_leaves': 128, 'min_child_samples': 60, 'subsample': 0.9, 'colsample_bytree': 0.85,
    'reg_alpha': 0.05, 'reg_lambda': 1.0, 'n_jobs': -1, 'verbose': -1,
}
m2_oof, m2_test, m2_cv = train_lgb_multiseed(X, y_log, X_test, folds, m2_params, sample_w, clip0_expm1, 'LGB_log1p_huber')
oof_preds['lgb_log1p_huber'] = m2_oof
test_preds['lgb_log1p_huber'] = m2_test
cv_maes['lgb_log1p_huber'] = m2_cv

# 모델 3: LightGBM sqrt+MAE
print("\n  [모델 3] LightGBM sqrt+MAE...", flush=True)
m3_params = {
    'objective': 'mae', 'n_estimators': 2000, 'learning_rate': 0.03,
    'num_leaves': 96, 'min_child_samples': 80, 'subsample': 0.9, 'colsample_bytree': 0.85,
    'reg_alpha': 0.1, 'reg_lambda': 1.5, 'n_jobs': -1, 'verbose': -1,
}
m3_oof, m3_test, m3_cv = train_lgb_multiseed(X, y_sqrt, X_test, folds, m3_params, sample_w, clip0_sq, 'LGB_sqrt_mae')
oof_preds['lgb_sqrt_mae'] = m3_oof
test_preds['lgb_sqrt_mae'] = m3_test
cv_maes['lgb_sqrt_mae'] = m3_cv

# 모델 4: XGBoost raw+MAE
print("\n  [모델 4] XGBoost raw+MAE...", flush=True)
m4_params = {
    'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 8,
    'min_child_weight': 6, 'subsample': 0.9, 'colsample_bytree': 0.85,
    'reg_lambda': 1.5, 'reg_alpha': 0.05,
    'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
    'tree_method': 'hist', 'verbosity': 0, 'early_stopping_rounds': 100,
}
m4_oof, m4_test, m4_cv = train_xgb_multiseed(X, y, X_test, folds, m4_params, sample_w, clip0, 'XGB_raw_mae')
oof_preds['xgb_raw_mae'] = m4_oof
test_preds['xgb_raw_mae'] = m4_test
cv_maes['xgb_raw_mae'] = m4_cv

# 모델 5: CatBoost log1p+MAE
print("\n  [모델 5] CatBoost log1p+MAE...", flush=True)
m5_params = {
    'iterations': 2000, 'learning_rate': 0.03, 'depth': 8,
    'l2_leaf_reg': 5.0, 'subsample': 0.9,
    'loss_function': 'MAE', 'verbose': 0,
}
m5_oof, m5_test, m5_cv = train_cat_multiseed(X, y_log, X_test, folds, m5_params, sample_w, clip0_expm1, 'Cat_log1p_mae')
oof_preds['cat_log1p_mae'] = m5_oof
test_preds['cat_log1p_mae'] = m5_test
cv_maes['cat_log1p_mae'] = m5_cv

# 모델 6: CatBoost raw+MAE
print("\n  [모델 6] CatBoost raw+MAE...", flush=True)
m6_params = {
    'iterations': 2000, 'learning_rate': 0.03, 'depth': 6,
    'l2_leaf_reg': 3.0, 'subsample': 0.85,
    'loss_function': 'MAE', 'verbose': 0,
}
m6_oof, m6_test, m6_cv = train_cat_multiseed(X, y, X_test, folds, m6_params, sample_w, clip0, 'Cat_raw_mae')
oof_preds['cat_raw_mae'] = m6_oof
test_preds['cat_raw_mae'] = m6_test
cv_maes['cat_raw_mae'] = m6_cv

# ============================================================
# 4. Level 1 가중 평균 앙상블
# ============================================================
print("\n=== Level 1 가중 평균 앙상블 ===", flush=True)

model_names = list(oof_preds.keys())
oof_matrix = np.column_stack([oof_preds[n] for n in model_names])
test_matrix = np.column_stack([test_preds[n] for n in model_names])

def l1_ensemble_mae(weights):
    w = weights / weights.sum()
    return mean_absolute_error(y, oof_matrix @ w)

result = minimize(
    l1_ensemble_mae,
    x0=np.ones(len(model_names)) / len(model_names),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)
l1_w = result.x / result.x.sum()
l1_oof = oof_matrix @ l1_w
l1_test = test_matrix @ l1_w
l1_cv = mean_absolute_error(y, l1_oof)

print("  최적 가중치:", flush=True)
for name, w in zip(model_names, l1_w):
    print(f"    {name:20s}: {w:.4f}", flush=True)
print(f"  Level 1 가중 평균 CV MAE: {l1_cv:.4f}", flush=True)

# ============================================================
# 5. Level 2 스태킹
# ============================================================
print("\n=== Level 2 LGB 스태킹 ===", flush=True)

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
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

meta_lgb_cv = mean_absolute_error(y, meta_lgb_oof)
print(f"  LGB 스태킹 CV MAE: {meta_lgb_cv:.4f}", flush=True)

# --- 최종 선택 ---
print("\n  비교:", flush=True)
print(f"    Level 1 가중 평균: {l1_cv:.4f}", flush=True)
print(f"    Level 2 LGB 스태킹: {meta_lgb_cv:.4f}", flush=True)

if meta_lgb_cv < l1_cv:
    final_cv = meta_lgb_cv
    final_test = meta_lgb_test
    final_method = 'lgb_stack'
else:
    final_cv = l1_cv
    final_test = l1_test
    final_method = 'l1_weighted'
print(f"  최종 선택: {final_method} (CV MAE: {final_cv:.4f})", flush=True)

# ============================================================
# 6. 결과 비교 테이블
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 9 결과 ===", flush=True)
print("=" * 60, flush=True)
print(f"Optuna 재튜닝: {optuna_best_mae:.4f} (기존 8.7494 → 신규 {optuna_best_mae:.4f})", flush=True)
print(f"Optuna best params: {optuna_best_params}", flush=True)
print(f"모델별 CV MAE (seed-averaged):", flush=True)
for name in model_names:
    print(f"  {name:20s}: {cv_maes[name]:.4f}", flush=True)
print(f"Level 1 가중 평균: {l1_cv:.4f}", flush=True)
print(f"Level 2 LGB 스태킹: {meta_lgb_cv:.4f}", flush=True)
print(f"Phase 8 대비 개선: {final_cv - 8.653:+.4f}", flush=True)

# ============================================================
# 7. 제출 파일
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
final_test = np.clip(final_test, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase9.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase9.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 8. 피처 중요도 (모델 1 기준)
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': m1_imp_model.feature_importances_
}).sort_values('importance', ascending=False)

top30 = importance.head(30).sort_values('importance')

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color='steelblue')
ax.set_title('Feature Importance Top 30 (Phase 9 - LGB raw+MAE Optuna)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('output/feature_importance_phase9.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase9.png 저장 완료", flush=True)

print("\n=== Phase 9 완료 ===", flush=True)
