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
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# ============================================================
# 1. 데이터 준비 (Phase 3B 로직 재사용)
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

# --- 시계열 피처 ---
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

# --- Phase 3B 인터랙션 피처 ---
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

base_feat_count = len([c for c in combined.columns if c not in
    ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']])
print(f"  기존 피처 수 (신규 추가 전): {base_feat_count}", flush=True)

# ============================================================
# 2. 신규 피처 — Onset 피처 (상태 변화 시작점 추적)
# ============================================================
print("\n=== Onset 피처 생성 ===", flush=True)
onset_features = []

# 충전 관련 onset
charging_flag = (combined['robot_charging'] > 0).astype(np.float32)
charging_flag_shifted = combined.groupby('scenario_id')[charging_flag.name if hasattr(charging_flag, 'name') else 'robot_charging'].transform(lambda x: x)
# 직접 계산
combined['_charging_flag'] = (combined['robot_charging'] > 0).astype(np.float32)
combined['_charging_flag_prev'] = combined.groupby('scenario_id')['_charging_flag'].shift(1).fillna(0)
combined['charging_ever_started'] = combined.groupby('scenario_id')['_charging_flag_prev'].cummax()

# charging_start_idx: 충전이 처음 발생한 슬롯 번호
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

print("  충전 onset 계산 중...", flush=True)
combined['charging_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_charging_flag'))
combined['charging_steps_since_start'] = np.where(
    combined['charging_start_idx'] >= 0,
    combined['implicit_timeslot'] - combined['charging_start_idx'],
    -1
).astype(np.float32)
combined['charging_started_early'] = (
    (combined['charging_start_idx'] >= 0) & (combined['charging_start_idx'] < 5)
).astype(np.float32)
onset_features += ['charging_ever_started', 'charging_start_idx',
                   'charging_steps_since_start', 'charging_started_early']

# 대기열 관련 onset
print("  대기열 onset 계산 중...", flush=True)
combined['_queue_flag'] = (combined['charge_queue_length'] > 0).astype(np.float32)
combined['_queue_flag_prev'] = combined.groupby('scenario_id')['_queue_flag'].shift(1).fillna(0)
combined['queue_ever_started'] = combined.groupby('scenario_id')['_queue_flag_prev'].cummax()
combined['queue_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_queue_flag'))
onset_features += ['queue_ever_started', 'queue_start_idx']

# 혼잡 관련 onset
print("  혼잡 onset 계산 중...", flush=True)
combined['_congestion_flag'] = (combined['congestion_score'] > 0).astype(np.float32)
combined['_congestion_flag_prev'] = combined.groupby('scenario_id')['_congestion_flag'].shift(1).fillna(0)
combined['congestion_ever_started'] = combined.groupby('scenario_id')['_congestion_flag_prev'].cummax()
combined['congestion_start_idx'] = combined.groupby('scenario_id', group_keys=False).apply(
    lambda g: compute_onset_idx(g, '_congestion_flag'))
onset_features += ['congestion_ever_started', 'congestion_start_idx']

# 임시 컬럼 삭제
combined.drop(columns=['_charging_flag', '_charging_flag_prev', '_queue_flag', '_queue_flag_prev',
                        '_congestion_flag', '_congestion_flag_prev'], inplace=True)

print(f"  Onset 피처 {len(onset_features)}개 생성 완료: {onset_features}", flush=True)

# ============================================================
# 3. 신규 피처 — Expanding Mean (누적 평균 + 이탈)
# ============================================================
print("\n=== Expanding Mean 피처 생성 ===", flush=True)
expanding_cols = [
    'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
    'heavy_item_ratio', 'robot_active', 'battery_mean', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'pack_utilization', 'loading_dock_util',
    'charge_queue_length', 'fault_count_15m', 'avg_trip_distance'
]
expanding_features = []

for i, col in enumerate(expanding_cols):
    shifted = combined.groupby('scenario_id')[col].shift(1)
    expmean = shifted.groupby(combined['scenario_id']).expanding().mean().droplevel(0)
    expmean = expmean.sort_index()
    combined[f'{col}_expmean_prev'] = expmean.astype(np.float32)
    combined[f'{col}_delta_expmean'] = (combined[col] - combined[f'{col}_expmean_prev']).astype(np.float32)
    expanding_features += [f'{col}_expmean_prev', f'{col}_delta_expmean']
    if (i + 1) % 5 == 0:
        print(f"  Expanding Mean 진행: {i+1}/{len(expanding_cols)}", flush=True)

print(f"  Expanding Mean 피처 {len(expanding_features)}개 생성 완료", flush=True)

# ============================================================
# 4. 신규 피처 — 비선형 임계값 피처
# ============================================================
print("\n=== 비선형 임계값 피처 생성 ===", flush=True)
nonlinear_features = []

combined['battery_mean_below_44'] = np.maximum(44.0 - combined['battery_mean'], 0).astype(np.float32)
combined['low_battery_ratio_above_02'] = np.maximum(combined['low_battery_ratio'] - 0.2, 0).astype(np.float32)
combined['pack_utilization_sq'] = (combined['pack_utilization'] ** 2).astype(np.float32)
combined['loading_dock_util_sq'] = (combined['loading_dock_util'] ** 2).astype(np.float32)
combined['congestion_score_sq'] = (combined['congestion_score'] ** 2).astype(np.float32)
combined['charge_pressure'] = ((combined['robot_charging'] + combined['charge_queue_length']) / (combined['charger_count'] + 1)).astype(np.float32)
combined['charge_pressure_sq'] = (combined['charge_pressure'] ** 2).astype(np.float32)

nonlinear_features = ['battery_mean_below_44', 'low_battery_ratio_above_02',
                       'pack_utilization_sq', 'loading_dock_util_sq', 'congestion_score_sq',
                       'charge_pressure', 'charge_pressure_sq']
print(f"  비선형 피처 {len(nonlinear_features)}개 생성 완료", flush=True)

# ============================================================
# 5. 신규 피처 — 시간대별 위상 피처
# ============================================================
print("\n=== 시간대별 위상 피처 생성 ===", flush=True)
phase_features = []

combined['is_early_phase'] = (combined['implicit_timeslot'] <= 5).astype(np.float32)
combined['is_mid_phase'] = ((combined['implicit_timeslot'] >= 6) & (combined['implicit_timeslot'] <= 15)).astype(np.float32)
combined['is_late_phase'] = (combined['implicit_timeslot'] >= 16).astype(np.float32)
combined['time_frac'] = (combined['implicit_timeslot'] / 24.0).astype(np.float32)
combined['time_remaining'] = (24 - combined['implicit_timeslot']).astype(np.float32)
combined['time_frac_sq'] = (combined['time_frac'] ** 2).astype(np.float32)

phase_features = ['is_early_phase', 'is_mid_phase', 'is_late_phase',
                   'time_frac', 'time_remaining', 'time_frac_sq']
print(f"  위상 피처 {len(phase_features)}개 생성 완료", flush=True)

# --- 신규 피처 요약 ---
all_new_features = onset_features + expanding_features + nonlinear_features + phase_features
print(f"\n  신규 피처 총계: onset {len(onset_features)}개 + expanding {len(expanding_features)}개 "
      f"+ 비선형 {len(nonlinear_features)}개 + 위상 {len(phase_features)}개 = {len(all_new_features)}개", flush=True)

# ============================================================
# float32 변환 + 분리
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
groups = train_fe['scenario_id']
time_idx = train_fe['implicit_timeslot'].values.astype(np.float32)
X_test = test_fe[feature_cols]

# ID 순서 검증
assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID 순서 불일치!"
print("ID 순서 검증 통과!", flush=True)

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y_log, groups))

# ============================================================
# 6. 샘플 가중치 구현
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
print(f"  가중치 분포: min={sample_w.min():.2f}, max={sample_w.max():.2f}, "
      f"mean={sample_w.mean():.2f}, median={np.median(sample_w):.2f}", flush=True)
print(f"  q90={np.nanquantile(y, 0.90):.2f}, q95={np.nanquantile(y, 0.95):.2f}, "
      f"q99={np.nanquantile(y, 0.99):.2f}", flush=True)

# ============================================================
# 가중치 효과 확인 (LGB raw+MAE, 가중치 없이 vs 가중치 적용)
# ============================================================
print("\n=== 샘플 가중치 효과 확인 ===", flush=True)

optuna_params = {
    'objective': 'mae',
    'n_estimators': 2000,
    'learning_rate': 0.0129,
    'num_leaves': 185,
    'max_depth': 9,
    'min_child_samples': 80,
    'reg_alpha': 0.0574,
    'reg_lambda': 0.0042,
    'feature_fraction': 0.6005,
    'bagging_fraction': 0.7663,
    'bagging_freq': 1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# 가중치 없이
print("  가중치 없이 (baseline)...", flush=True)
oof_no_w = np.zeros(len(X))
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(**optuna_params)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_no_w[va_idx] = np.clip(m.predict(X.iloc[va_idx]), 0, None)
mae_no_w = mean_absolute_error(y, oof_no_w)
print(f"  가중치 없이 CV MAE: {mae_no_w:.4f}", flush=True)

# 가중치 적용
print("  가중치 적용...", flush=True)
oof_w = np.zeros(len(X))
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(**optuna_params)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sample_w[tr_idx],
          eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_w[va_idx] = np.clip(m.predict(X.iloc[va_idx]), 0, None)
mae_w = mean_absolute_error(y, oof_w)
print(f"  가중치 적용 CV MAE: {mae_w:.4f}", flush=True)
print(f"  가중치 효과: {mae_w - mae_no_w:+.4f}", flush=True)

# ============================================================
# 7. 4모델 앙상블 — 다양한 변환+목적함수
# ============================================================
print("\n=== 4모델 앙상블 학습 ===", flush=True)

# --- 모델 1: LightGBM + 변환 없음 + objective='mae' ---
print("\n  [모델 1] LightGBM raw+MAE (Optuna params + sample_weight)...", flush=True)
m1_params = {
    'objective': 'mae',
    'n_estimators': 2000,
    'learning_rate': 0.0129,
    'num_leaves': 185,
    'max_depth': 9,
    'min_child_samples': 80,
    'reg_alpha': 0.0574,
    'reg_lambda': 0.0042,
    'feature_fraction': 0.6005,
    'bagging_fraction': 0.7663,
    'bagging_freq': 1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}
m1_oof = np.zeros(len(X))
m1_test = np.zeros(len(X_test))
m1_maes = []
m1_models = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    m = lgb.LGBMRegressor(**m1_params)
    m.fit(Xtr, ytr, sample_weight=sample_w[tr_idx],
          eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(m.predict(Xva), 0, None)
    m1_oof[va_idx] = pred
    m1_test += np.clip(m.predict(X_test), 0, None) / 5
    mae = mean_absolute_error(yva, pred)
    m1_maes.append(mae)
    m1_models.append(m)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

m1_cv = mean_absolute_error(y, m1_oof)
print(f"  모델 1 CV MAE: {m1_cv:.4f}", flush=True)

# --- 모델 2: LightGBM + log1p + objective='huber' ---
print("\n  [모델 2] LightGBM log1p+Huber (sample_weight)...", flush=True)
m2_params = {
    'objective': 'huber',
    'huber_delta': 0.9,
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'num_leaves': 128,
    'min_child_samples': 60,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.05,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}
m2_oof = np.zeros(len(X))
m2_test = np.zeros(len(X_test))
m2_maes = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr_log, yva_log = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = lgb.LGBMRegressor(**m2_params)
    m.fit(Xtr, ytr_log, sample_weight=sample_w[tr_idx],
          eval_set=[(Xva, yva_log)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m.predict(Xva)), 0, None)
    m2_oof[va_idx] = pred
    m2_test += np.clip(np.expm1(m.predict(X_test)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    m2_maes.append(mae)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

m2_cv = mean_absolute_error(y, m2_oof)
print(f"  모델 2 CV MAE: {m2_cv:.4f}", flush=True)

# --- 모델 3: XGBoost + 변환 없음 + objective='reg:absoluteerror' ---
print("\n  [모델 3] XGBoost raw+MAE (sample_weight)...", flush=True)
m3_oof = np.zeros(len(X))
m3_test = np.zeros(len(X_test))
m3_maes = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    m = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        min_child_weight=6, subsample=0.9, colsample_bytree=0.85,
        reg_lambda=1.5, reg_alpha=0.05,
        objective='reg:absoluteerror', eval_metric='mae',
        tree_method='hist', random_state=42, verbosity=0,
        early_stopping_rounds=100
    )
    m.fit(Xtr, ytr, sample_weight=sample_w[tr_idx],
          eval_set=[(Xva, yva)], verbose=False)
    pred = np.clip(m.predict(Xva), 0, None)
    m3_oof[va_idx] = pred
    m3_test += np.clip(m.predict(X_test), 0, None) / 5
    mae = mean_absolute_error(yva, pred)
    m3_maes.append(mae)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration})", flush=True)

m3_cv = mean_absolute_error(y, m3_oof)
print(f"  모델 3 CV MAE: {m3_cv:.4f}", flush=True)

# --- 모델 4: CatBoost + log1p + loss_function='MAE' ---
print("\n  [모델 4] CatBoost log1p+MAE (sample_weight)...", flush=True)
m4_oof = np.zeros(len(X))
m4_test = np.zeros(len(X_test))
m4_maes = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr_log, yva_log = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    train_pool = Pool(Xtr, ytr_log, weight=sample_w[tr_idx])
    eval_pool = Pool(Xva, yva_log)
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=8,
        l2_leaf_reg=5.0, subsample=0.9,
        loss_function='MAE', random_seed=42, verbose=0
    )
    m.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=100)
    pred = np.clip(np.expm1(m.predict(Xva)), 0, None)
    m4_oof[va_idx] = pred
    m4_test += np.clip(np.expm1(m.predict(X_test)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    m4_maes.append(mae)
    print(f"    Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

m4_cv = mean_absolute_error(y, m4_oof)
print(f"  모델 4 CV MAE: {m4_cv:.4f}", flush=True)

# ============================================================
# 8. 앙상블 가중치 최적화
# ============================================================
print("\n=== 앙상블 가중치 최적화 ===", flush=True)

def ensemble_mae(weights):
    w = weights / weights.sum()
    blended = w[0] * m1_oof + w[1] * m2_oof + w[2] * m3_oof + w[3] * m4_oof
    return mean_absolute_error(y, blended)

result = minimize(
    ensemble_mae,
    x0=np.array([0.25, 0.25, 0.25, 0.25]),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)

best_w = result.x / result.x.sum()
ensemble_oof = best_w[0] * m1_oof + best_w[1] * m2_oof + best_w[2] * m3_oof + best_w[3] * m4_oof
ens_cv = mean_absolute_error(y, ensemble_oof)

print(f"  최적 가중치 - lgb_raw={best_w[0]:.4f}, lgb_huber={best_w[1]:.4f}, "
      f"xgb={best_w[2]:.4f}, cat={best_w[3]:.4f}", flush=True)
print(f"  앙상블 CV MAE: {ens_cv:.4f}", flush=True)

final_test = best_w[0] * m1_test + best_w[1] * m2_test + best_w[2] * m3_test + best_w[3] * m4_test

# ============================================================
# 9. 결과 비교 테이블
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 7 결과 ===", flush=True)
print("=" * 60, flush=True)

# 피처 중요도 Top 30 내 신규 피처 수
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': m1_models[0].feature_importances_
}).sort_values('importance', ascending=False)
top30_features = importance.head(30)['feature'].tolist()
new_in_top30 = [f for f in all_new_features if f in top30_features]

print(f"신규 피처 수: onset {len(onset_features)}개 + expanding {len(expanding_features)}개 "
      f"+ 비선형 {len(nonlinear_features)}개 + 위상 {len(phase_features)}개 = {len(all_new_features)}개", flush=True)
print(f"총 피처 수: {len(feature_cols)}개", flush=True)
print(f"모델별 CV MAE:", flush=True)
print(f"  LGB raw+MAE:        {m1_cv:.4f}", flush=True)
print(f"  LGB log1p+Huber:    {m2_cv:.4f}", flush=True)
print(f"  XGB raw+MAE:        {m3_cv:.4f}", flush=True)
print(f"  CatBoost log1p+MAE: {m4_cv:.4f}", flush=True)
print(f"앙상블 가중치: lgb_raw={best_w[0]:.4f}, lgb_huber={best_w[1]:.4f}, "
      f"xgb={best_w[2]:.4f}, cat={best_w[3]:.4f}", flush=True)
print(f"앙상블 CV MAE: {ens_cv:.4f}", flush=True)
print(f"Phase 3B 대비 개선: {ens_cv - 8.786:+.4f}", flush=True)
print(f"샘플 가중치 효과:", flush=True)
print(f"  가중치 적용 전: {mae_no_w:.4f}", flush=True)
print(f"  가중치 적용 후: {mae_w:.4f}", flush=True)
print(f"  효과: {mae_w - mae_no_w:+.4f}", flush=True)

# ============================================================
# 10. 제출 파일 생성
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
final_test = np.clip(final_test, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase7.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase7.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 11. 피처 중요도 시각화
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)

top30 = importance.head(30).sort_values('importance')
ts_suffixes = ['_lag', '_roll', '_diff1', '_cumsum']
phase3b_features = ['orders_per_packstation', 'pack_dock_pressure', 'dock_wait_pressure',
                    'shift_load_pressure', 'battery_congestion', 'storage_density_congestion',
                    'battery_trip_pressure', 'demand_density']

def get_color(f):
    if f in all_new_features:
        return 'darkgreen'
    elif f in phase3b_features:
        return 'forestgreen'
    elif any(s in f for s in ts_suffixes):
        return 'coral'
    else:
        return 'steelblue'

colors = [get_color(f) for f in top30['feature']]

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color=colors)
ax.set_title('피처 중요도 Top 30 (Phase 7 - LGB raw+MAE)')
ax.set_xlabel('중요도')
ax.legend(handles=[
    Patch(color='coral', label='시계열 피처'),
    Patch(color='steelblue', label='기존 피처'),
    Patch(color='forestgreen', label='Phase 3B 인터랙션'),
    Patch(color='darkgreen', label='Phase 7 신규 피처')
])
plt.tight_layout()
plt.savefig('output/feature_importance_phase7.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase7.png 저장 완료", flush=True)

print(f"\n신규 피처 중 중요도 Top 30 진입: {len(new_in_top30)}개", flush=True)
if new_in_top30:
    for f in new_in_top30:
        rank = top30_features.index(f) + 1
        print(f"  #{rank:2d} {f}", flush=True)

print("\n=== Phase 7 완료 ===", flush=True)
