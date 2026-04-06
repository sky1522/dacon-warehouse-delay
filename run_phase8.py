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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# ============================================================
# 1. 데이터 준비 (Phase 7 로직 재사용)
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

# --- Phase 3B 인터랙션 피처 (8개) ---
print("=== Phase 3B 인터랙션 피처 추가 ===", flush=True)
combined['orders_per_packstation'] = combined['order_inflow_15m'] / combined['pack_station_count'].replace(0, np.nan)
combined['pack_dock_pressure'] = combined['pack_utilization'] * combined['loading_dock_util']
combined['dock_wait_pressure'] = combined['outbound_truck_wait_min'] * combined['loading_dock_util']
combined['shift_load_pressure'] = combined['prev_shift_volume'] * combined['order_inflow_15m']
combined['battery_congestion'] = combined['low_battery_ratio'] * combined['congestion_score']
combined['storage_density_congestion'] = combined['storage_density_pct'] * combined['congestion_score']
combined['battery_trip_pressure'] = combined['low_battery_ratio'] * combined['avg_trip_distance']
combined['demand_density'] = combined['order_inflow_15m'] * combined['max_zone_density']
print("  인터랙션 피처 8개 추가 완료", flush=True)

# --- 결측 지시자 ---
train_part = combined[combined['_is_train'] == 1]
missing_counts = train_part.isnull().sum().sort_values(ascending=False)
top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith('_')]
for col in top10_missing:
    if col in combined.columns:
        combined[f'{col}_missing'] = combined[col].isnull().astype(int)
del train_part
gc.collect()

# --- Phase 7: Onset 피처 (8개) ---
print("\n=== Onset 피처 생성 ===", flush=True)
onset_features = []

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

# 충전 onset
combined['_charging_flag'] = (combined['robot_charging'] > 0).astype(np.float32)
combined['_charging_flag_prev'] = combined.groupby('scenario_id')['_charging_flag'].shift(1).fillna(0)
combined['charging_ever_started'] = combined.groupby('scenario_id')['_charging_flag_prev'].cummax()
print("  충전 onset 계산 중...", flush=True)
combined['charging_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_charging_flag'))
combined['charging_steps_since_start'] = np.where(
    combined['charging_start_idx'] >= 0,
    combined['implicit_timeslot'] - combined['charging_start_idx'], -1).astype(np.float32)
combined['charging_started_early'] = (
    (combined['charging_start_idx'] >= 0) & (combined['charging_start_idx'] < 5)).astype(np.float32)
onset_features += ['charging_ever_started', 'charging_start_idx',
                   'charging_steps_since_start', 'charging_started_early']

# 대기열 onset
print("  대기열 onset 계산 중...", flush=True)
combined['_queue_flag'] = (combined['charge_queue_length'] > 0).astype(np.float32)
combined['_queue_flag_prev'] = combined.groupby('scenario_id')['_queue_flag'].shift(1).fillna(0)
combined['queue_ever_started'] = combined.groupby('scenario_id')['_queue_flag_prev'].cummax()
combined['queue_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_queue_flag'))
onset_features += ['queue_ever_started', 'queue_start_idx']

# 혼잡 onset
print("  혼잡 onset 계산 중...", flush=True)
combined['_congestion_flag'] = (combined['congestion_score'] > 0).astype(np.float32)
combined['_congestion_flag_prev'] = combined.groupby('scenario_id')['_congestion_flag'].shift(1).fillna(0)
combined['congestion_ever_started'] = combined.groupby('scenario_id')['_congestion_flag_prev'].cummax()
combined['congestion_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_congestion_flag'))
onset_features += ['congestion_ever_started', 'congestion_start_idx']

combined.drop(columns=['_charging_flag', '_charging_flag_prev', '_queue_flag', '_queue_flag_prev',
                        '_congestion_flag', '_congestion_flag_prev'], inplace=True)
print(f"  Onset 피처 {len(onset_features)}개 생성 완료", flush=True)

# --- Phase 7: Expanding Mean (30개) ---
print("\n=== Expanding Mean 피처 생성 ===", flush=True)
expanding_cols_mean = [
    'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
    'heavy_item_ratio', 'robot_active', 'battery_mean', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'pack_utilization', 'loading_dock_util',
    'charge_queue_length', 'fault_count_15m', 'avg_trip_distance'
]
expanding_features = []

for i, col in enumerate(expanding_cols_mean):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    expmean = shifted.groupby(combined['scenario_id']).expanding().mean().droplevel(0).sort_index()
    combined[f'{col}_expmean_prev'] = expmean.astype(np.float32)
    combined[f'{col}_delta_expmean'] = (combined[col] - combined[f'{col}_expmean_prev']).astype(np.float32)
    expanding_features += [f'{col}_expmean_prev', f'{col}_delta_expmean']
    if (i + 1) % 5 == 0:
        print(f"  Expanding Mean 진행: {i+1}/{len(expanding_cols_mean)}", flush=True)

print(f"  Expanding Mean 피처 {len(expanding_features)}개 생성 완료", flush=True)

# --- Phase 7: 비선형 (7개) ---
print("\n=== 비선형 임계값 피처 생성 ===", flush=True)
combined['battery_mean_below_44'] = np.maximum(44.0 - combined['battery_mean'], 0).astype(np.float32)
combined['low_battery_ratio_above_02'] = np.maximum(combined['low_battery_ratio'] - 0.2, 0).astype(np.float32)
combined['pack_utilization_sq'] = (combined['pack_utilization'] ** 2).astype(np.float32)
combined['loading_dock_util_sq'] = (combined['loading_dock_util'] ** 2).astype(np.float32)
combined['congestion_score_sq'] = (combined['congestion_score'] ** 2).astype(np.float32)
combined['charge_pressure_nl'] = ((combined['robot_charging'] + combined['charge_queue_length']) / (combined['charger_count'] + 1)).astype(np.float32)
combined['charge_pressure_nl_sq'] = (combined['charge_pressure_nl'] ** 2).astype(np.float32)
nonlinear_features = ['battery_mean_below_44', 'low_battery_ratio_above_02',
                       'pack_utilization_sq', 'loading_dock_util_sq', 'congestion_score_sq',
                       'charge_pressure_nl', 'charge_pressure_nl_sq']
print(f"  비선형 피처 {len(nonlinear_features)}개 생성 완료", flush=True)

# --- Phase 7: 위상 (6개) ---
print("\n=== 시간대별 위상 피처 생성 ===", flush=True)
combined['is_early_phase'] = (combined['implicit_timeslot'] <= 5).astype(np.float32)
combined['is_mid_phase'] = ((combined['implicit_timeslot'] >= 6) & (combined['implicit_timeslot'] <= 15)).astype(np.float32)
combined['is_late_phase'] = (combined['implicit_timeslot'] >= 16).astype(np.float32)
combined['time_frac'] = (combined['implicit_timeslot'] / 24.0).astype(np.float32)
combined['time_remaining'] = (24 - combined['implicit_timeslot']).astype(np.float32)
combined['time_frac_sq'] = (combined['time_frac'] ** 2).astype(np.float32)
phase_features = ['is_early_phase', 'is_mid_phase', 'is_late_phase',
                   'time_frac', 'time_remaining', 'time_frac_sq']
print(f"  위상 피처 {len(phase_features)}개 생성 완료", flush=True)

phase7_new = onset_features + expanding_features + nonlinear_features + phase_features
print(f"\n  Phase 7 신규 피처 합계: {len(phase7_new)}개", flush=True)

# ============================================================
# 2. 신규 피처 A: 경쟁자 코드에서 가져올 피처
# ============================================================
print("\n=== 경쟁자 피처 추가 ===", flush=True)
competitor_features = []

# A-1) 로봇 상태 분해 (6개)
print("  A-1) 로봇 상태 분해 (6개)...", flush=True)
combined['robot_total_state'] = combined['robot_active'] + combined['robot_idle'] + combined['robot_charging']
combined['robot_total_gap'] = combined['robot_total_state'] - combined['robot_total']
combined['robot_active_share'] = combined['robot_active'] / (combined['robot_total_state'] + 1)
combined['robot_idle_share'] = combined['robot_idle'] / (combined['robot_total_state'] + 1)
combined['robot_charging_share'] = combined['robot_charging'] / (combined['robot_total_state'] + 1)
combined['charging_to_active_ratio'] = combined['robot_charging'] / (combined['robot_active'] + 1)
competitor_features += ['robot_total_state', 'robot_total_gap', 'robot_active_share',
                        'robot_idle_share', 'robot_charging_share', 'charging_to_active_ratio']

# A-2) 수요-용량 비율 (15개)
print("  A-2) 수요-용량 비율 (15개)...", flush=True)
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
competitor_features += ['inflow_per_robot', 'inflow_per_pack_station', 'unique_sku_per_robot',
                        'unique_sku_per_pack_station', 'charge_queue_per_charger', 'charging_per_charger',
                        'congestion_per_width', 'zone_density_per_width', 'order_per_sqm',
                        'dock_pressure', 'fault_per_active', 'collision_per_active',
                        'blocked_per_active', 'congestion_per_active', 'label_queue_per_pack']

# A-3) 복합 인터랙션 (12개)
print("  A-3) 복합 인터랙션 (12개)...", flush=True)
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
competitor_features += ['demand_mass', 'demand_mass_per_robot', 'trip_load', 'trip_load_per_robot',
                        'complexity_load', 'complexity_load_per_pack', 'congestion_x_lowbat',
                        'battery_pressure', 'queue_wait_pressure', 'dock_pack_pressure',
                        'staging_pack_pressure', 'charge_pressure']

# A-4) layout 밀도 피처 (8개)
print("  A-4) layout 밀도 피처 (8개)...", flush=True)
combined['warehouse_volume'] = combined['floor_area_sqm'] * combined['ceiling_height_m']
combined['intersection_density'] = combined['intersection_count'] / (combined['floor_area_sqm'] + 1)
combined['pack_station_density'] = combined['pack_station_count'] / (combined['floor_area_sqm'] + 1)
combined['charger_density'] = combined['charger_count'] / (combined['floor_area_sqm'] + 1)
combined['robot_density_layout'] = combined['robot_total'] / (combined['floor_area_sqm'] + 1)
combined['movement_friction'] = combined['intersection_count'] / (combined['aisle_width_avg'] + 0.01)
combined['layout_compact_x_dispersion'] = combined['layout_compactness'] * combined['zone_dispersion']
combined['one_way_friction'] = combined['one_way_ratio'] * combined['intersection_count'] / (combined['aisle_width_avg'] + 0.01)
competitor_features += ['warehouse_volume', 'intersection_density', 'pack_station_density',
                        'charger_density', 'robot_density_layout', 'movement_friction',
                        'layout_compact_x_dispersion', 'one_way_friction']

# A-5) missing indicator (3개)
print("  A-5) missing indicator (3개)...", flush=True)
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
competitor_features += ['n_missing_all', 'n_missing_dynamic', 'missing_ratio']

# A-6) rolling max + 편차 (10개)
print("  A-6) rolling max + 편차 (10개)...", flush=True)
rollmax_cols = ['order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization', 'loading_dock_util']
for col in rollmax_cols:
    shifted = combined.groupby('scenario_id')[col].shift(1)
    rollmax3 = shifted.groupby(combined['scenario_id']).rolling(3, min_periods=1).max().droplevel(0).sort_index()
    combined[f'{col}_rollmax3_prev'] = rollmax3.astype(np.float32)
    combined[f'{col}_dev_rollmax3'] = (combined[col] - combined[f'{col}_rollmax3_prev']).astype(np.float32)
    competitor_features += [f'{col}_rollmax3_prev', f'{col}_dev_rollmax3']

print(f"  경쟁자 피처 총 {len(competitor_features)}개 추가 완료", flush=True)

# ============================================================
# 3. 신규 피처 B: Expanding 확장 (std, max — 20개)
# ============================================================
print("\n=== Expanding 확장 (std, max) 생성 ===", flush=True)
expanding_ext_cols = [
    'order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization',
    'loading_dock_util', 'robot_active', 'low_battery_ratio', 'avg_trip_distance',
    'unique_sku_15m', 'max_zone_density'
]
expanding_ext_features = []

for i, col in enumerate(expanding_ext_cols):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    grp = shifted.groupby(combined['scenario_id'])
    expstd = grp.expanding().std().droplevel(0).sort_index()
    expmax = grp.expanding().max().droplevel(0).sort_index()
    combined[f'{col}_expstd_prev'] = expstd.astype(np.float32)
    combined[f'{col}_expmax_prev'] = expmax.astype(np.float32)
    expanding_ext_features += [f'{col}_expstd_prev', f'{col}_expmax_prev']
    if (i + 1) % 5 == 0:
        print(f"  Expanding 확장 진행: {i+1}/{len(expanding_ext_cols)}", flush=True)

print(f"  Expanding 확장 피처 {len(expanding_ext_features)}개 생성 완료", flush=True)

# --- 신규 피처 요약 ---
all_new_phase8 = competitor_features + expanding_ext_features
print(f"\n  Phase 8 신규 피처: 경쟁자 {len(competitor_features)}개 + Expanding확장 {len(expanding_ext_features)}개 = {len(all_new_phase8)}개", flush=True)

# ============================================================
# 4. float32 변환 + 분리
# ============================================================
print("\n=== float32 변환 + 데이터 분리 ===", flush=True)
float64_cols = combined.select_dtypes(include='float64').columns
combined[float64_cols] = combined[float64_cols].astype(np.float32)
print(f"  float64 → float32 변환: {len(float64_cols)}개 컬럼", flush=True)

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

# ============================================================
# 5. 샘플 가중치
# ============================================================
print("\n=== 샘플 가중치 구현 ===", flush=True)

def build_sample_weight(y, time_idx):
    w = np.ones(len(y), dtype=np.float32)
    q90 = np.nanquantile(y, 0.90)
    q95 = np.nanquantile(y, 0.95)
    q99 = np.nanquantile(y, 0.99)
    w += 0.15 * (y >= q90).astype(np.float32)
    w += 0.30 * (y >= q95).astype(np.float32)
    w += 0.60 * (y >= q99).astype(np.float32)
    if time_idx is not None:
        w += 0.08 * (time_idx / 24.0).astype(np.float32)
    return w

sample_w = build_sample_weight(y.values, time_idx)
print(f"  가중치 분포: min={sample_w.min():.2f}, max={sample_w.max():.2f}, mean={sample_w.mean():.2f}", flush=True)

# ============================================================
# 6. Level 1: 6모델 학습
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Level 1: 6모델 학습 ===", flush=True)
print("=" * 60, flush=True)

oof_preds = {}  # model_name -> oof array
test_preds = {}  # model_name -> test array
cv_maes = {}

# --- 모델 1: LightGBM raw+MAE (Optuna params) ---
print("\n  [모델 1] LightGBM raw+MAE (Optuna params)...", flush=True)
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
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    m1_models.append(m)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

m1_cv = mean_absolute_error(y, m1_oof)
oof_preds['lgb_raw_mae'] = m1_oof
test_preds['lgb_raw_mae'] = m1_test
cv_maes['lgb_raw_mae'] = m1_cv
print(f"  모델 1 CV MAE: {m1_cv:.4f}", flush=True)

# --- 모델 2: LightGBM log1p+Huber ---
print("\n  [모델 2] LightGBM log1p+Huber...", flush=True)
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
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

m2_cv = mean_absolute_error(y, m2_oof)
oof_preds['lgb_log1p_huber'] = m2_oof
test_preds['lgb_log1p_huber'] = m2_test
cv_maes['lgb_log1p_huber'] = m2_cv
print(f"  모델 2 CV MAE: {m2_cv:.4f}", flush=True)

# --- 모델 3: LightGBM sqrt+MAE ---
print("\n  [모델 3] LightGBM sqrt+MAE...", flush=True)
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
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

m3_cv = mean_absolute_error(y, m3_oof)
oof_preds['lgb_sqrt_mae'] = m3_oof
test_preds['lgb_sqrt_mae'] = m3_test
cv_maes['lgb_sqrt_mae'] = m3_cv
print(f"  모델 3 CV MAE: {m3_cv:.4f}", flush=True)

# --- 모델 4: XGBoost raw+MAE ---
print("\n  [모델 4] XGBoost raw+MAE...", flush=True)
m4_oof = np.zeros(len(X), dtype=np.float32)
m4_test = np.zeros(len(X_test), dtype=np.float32)

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        min_child_weight=6, subsample=0.9, colsample_bytree=0.85,
        reg_lambda=1.5, reg_alpha=0.05,
        objective='reg:absoluteerror', eval_metric='mae',
        tree_method='hist', random_state=42, verbosity=0,
        early_stopping_rounds=100)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sample_w[tr_idx],
          eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
    pred = np.clip(m.predict(X.iloc[va_idx]), 0, None).astype(np.float32)
    m4_oof[va_idx] = pred
    m4_test += np.clip(m.predict(X_test), 0, None).astype(np.float32) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration})", flush=True)

m4_cv = mean_absolute_error(y, m4_oof)
oof_preds['xgb_raw_mae'] = m4_oof
test_preds['xgb_raw_mae'] = m4_test
cv_maes['xgb_raw_mae'] = m4_cv
print(f"  모델 4 CV MAE: {m4_cv:.4f}", flush=True)

# --- 모델 5: CatBoost log1p+MAE ---
print("\n  [모델 5] CatBoost log1p+MAE...", flush=True)
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
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

m5_cv = mean_absolute_error(y, m5_oof)
oof_preds['cat_log1p_mae'] = m5_oof
test_preds['cat_log1p_mae'] = m5_test
cv_maes['cat_log1p_mae'] = m5_cv
print(f"  모델 5 CV MAE: {m5_cv:.4f}", flush=True)

# --- 모델 6: CatBoost raw+MAE ---
print("\n  [모델 6] CatBoost raw+MAE...", flush=True)
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
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

m6_cv = mean_absolute_error(y, m6_oof)
oof_preds['cat_raw_mae'] = m6_oof
test_preds['cat_raw_mae'] = m6_test
cv_maes['cat_raw_mae'] = m6_cv
print(f"  모델 6 CV MAE: {m6_cv:.4f}", flush=True)

# ============================================================
# 7. Level 1 앙상블 (가중 평균 — 비교 기준)
# ============================================================
print("\n=== Level 1 가중 평균 앙상블 ===", flush=True)

model_names = list(oof_preds.keys())
oof_matrix = np.column_stack([oof_preds[n] for n in model_names])
test_matrix = np.column_stack([test_preds[n] for n in model_names])

def l1_ensemble_mae(weights):
    w = weights / weights.sum()
    blended = oof_matrix @ w
    return mean_absolute_error(y, blended)

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
# 8. Level 2: 스태킹 메타 모델
# ============================================================
print("\n=== Level 2: 스태킹 메타 모델 ===", flush=True)

# 메타 피처 구성
meta_cols_extra = ['implicit_timeslot', 'order_inflow_15m', 'battery_mean',
                   'robot_active', 'pack_utilization', 'congestion_score']
meta_train_extra = train_fe[meta_cols_extra].values.astype(np.float32)
meta_test_extra = test_fe[meta_cols_extra].values.astype(np.float32)

meta_X_train = np.column_stack([oof_matrix, meta_train_extra])
meta_X_test = np.column_stack([test_matrix, meta_test_extra])
meta_feature_names = model_names + meta_cols_extra
print(f"  메타 피처 수: {len(meta_feature_names)} ({len(model_names)} OOF + {len(meta_cols_extra)} 원본)", flush=True)

meta_groups = groups.values

# --- Ridge ---
print("\n  [메타 모델 1] Ridge Regression...", flush=True)
ridge_oof = np.zeros(len(X), dtype=np.float32)
ridge_test = np.zeros(len(meta_X_test), dtype=np.float32)

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = Ridge(alpha=1.0)
    m.fit(meta_X_train[tr_idx], y.values[tr_idx])
    pred = np.clip(m.predict(meta_X_train[va_idx]), 0, None).astype(np.float32)
    ridge_oof[va_idx] = pred
    ridge_test += np.clip(m.predict(meta_X_test), 0, None).astype(np.float32) / 5
    mae = mean_absolute_error(y.values[va_idx], pred)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f}", flush=True)

ridge_cv = mean_absolute_error(y, ridge_oof)
print(f"  Ridge 스태킹 CV MAE: {ridge_cv:.4f}", flush=True)

# --- LightGBM 소규모 ---
print("\n  [메타 모델 2] LightGBM (소규모)...", flush=True)
meta_lgb_oof = np.zeros(len(X), dtype=np.float32)
meta_lgb_test = np.zeros(len(meta_X_test), dtype=np.float32)

meta_X_train_df = pd.DataFrame(meta_X_train, columns=meta_feature_names)
meta_X_test_df = pd.DataFrame(meta_X_test, columns=meta_feature_names)

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

# --- 최적 메타 모델 선택 ---
print("\n  스태킹 비교:", flush=True)
print(f"    Level 1 가중 평균: {l1_cv:.4f}", flush=True)
print(f"    Ridge 스태킹:     {ridge_cv:.4f}", flush=True)
print(f"    LGB 스태킹:       {meta_lgb_cv:.4f}", flush=True)

best_options = {
    'l1_weighted': (l1_cv, l1_test),
    'ridge_stack': (ridge_cv, ridge_test),
    'lgb_stack': (meta_lgb_cv, meta_lgb_test),
}
best_name = min(best_options, key=lambda k: best_options[k][0])
final_cv = best_options[best_name][0]
final_test = best_options[best_name][1]
print(f"\n  최종 선택: {best_name} (CV MAE: {final_cv:.4f})", flush=True)

# ============================================================
# 9. 결과 비교 테이블
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 8 결과 ===", flush=True)
print("=" * 60, flush=True)

# 피처 중요도
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': m1_models[0].feature_importances_
}).sort_values('importance', ascending=False)
top30_features = importance.head(30)['feature'].tolist()
comp_in_top30 = [f for f in all_new_phase8 if f in top30_features]

print(f"신규 피처: 경쟁자 A({len(competitor_features)}개) + Expanding 확장 B({len(expanding_ext_features)}개) = {len(all_new_phase8)}개", flush=True)
print(f"총 피처 수: {len(feature_cols)}개", flush=True)
print(f"Level 1 모델별 CV MAE:", flush=True)
for name in model_names:
    print(f"  {name:20s}: {cv_maes[name]:.4f}", flush=True)
print(f"Level 1 가중 평균 앙상블: {l1_cv:.4f}", flush=True)
print(f"Level 2 Ridge 스태킹:    {ridge_cv:.4f}", flush=True)
print(f"Level 2 LGB 스태킹:      {meta_lgb_cv:.4f}", flush=True)
print(f"최종 선택:               {best_name} ({final_cv:.4f})", flush=True)
print(f"Phase 7 대비 개선:       {final_cv - 8.675:+.4f}", flush=True)

# ============================================================
# 10. 제출 파일 생성
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
final_test = np.clip(final_test, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase8.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase8.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 11. 피처 중요도 시각화
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)

top30 = importance.head(30).sort_values('importance')
ts_suffixes = ['_lag', '_roll', '_diff1', '_cumsum']
phase3b_features_list = ['orders_per_packstation', 'pack_dock_pressure', 'dock_wait_pressure',
                         'shift_load_pressure', 'battery_congestion', 'storage_density_congestion',
                         'battery_trip_pressure', 'demand_density']

def get_color(f):
    if f in all_new_phase8:
        return 'red'
    elif f in phase7_new:
        return 'darkgreen'
    elif f in phase3b_features_list:
        return 'forestgreen'
    elif any(s in f for s in ts_suffixes):
        return 'coral'
    else:
        return 'steelblue'

colors = [get_color(f) for f in top30['feature']]

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color=colors)
ax.set_title('Feature Importance Top 30 (Phase 8 - LGB raw+MAE)')
ax.set_xlabel('Importance')
ax.legend(handles=[
    Patch(color='coral', label='Time-series'),
    Patch(color='steelblue', label='Base'),
    Patch(color='forestgreen', label='Phase 3B/7'),
    Patch(color='red', label='Phase 8 New')
])
plt.tight_layout()
plt.savefig('output/feature_importance_phase8.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase8.png 저장 완료", flush=True)

print(f"\n경쟁자 피처 중 중요도 Top 30 진입: {len(comp_in_top30)}개", flush=True)
if comp_in_top30:
    for f in comp_in_top30:
        rank = top30_features.index(f) + 1
        print(f"  #{rank:2d} {f}", flush=True)

print("\n=== Phase 8 완료 ===", flush=True)
