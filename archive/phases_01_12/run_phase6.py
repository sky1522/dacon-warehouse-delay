import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from matplotlib.patches import Patch
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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
print("  Phase 3B 인터랙션 피처 8개 추가 완료", flush=True)

# --- 결측 지시자 ---
train_part = combined[combined['_is_train'] == 1]
missing_counts = train_part.isnull().sum().sort_values(ascending=False)
top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith('_')]
for col in top10_missing:
    if col in combined.columns:
        combined[f'{col}_missing'] = combined[col].isnull().astype(int)

print(f"  기존 피처 수 (도메인 피처 추가 전): {len([c for c in combined.columns if c not in ['ID','layout_id','scenario_id','layout_type','avg_delay_minutes_next_30m','_is_train','_original_idx']])}", flush=True)

# ============================================================
# 2. 도메인 기반 신규 피처 추가
# ============================================================
print("\n=== 도메인 기반 신규 피처 추가 ===", flush=True)
domain_features = []

# A) 체인 병목 감지 피처
print("  A) 체인 병목 감지 피처...", flush=True)
combined['picking_packing_gap'] = combined['robot_utilization'] - combined['pack_utilization']
combined['packing_shipping_gap'] = combined['pack_utilization'] - combined['loading_dock_util']
combined['chain_pressure'] = combined['order_inflow_15m'] / (combined['pack_station_count'] * (combined['loading_dock_util'] + 0.01))
combined['picking_bottleneck'] = combined['robot_utilization'] * (1 - combined['pack_utilization'])
combined['shipping_bottleneck'] = combined['pack_utilization'] * combined['loading_dock_util']
domain_features += ['picking_packing_gap', 'packing_shipping_gap', 'chain_pressure',
                    'picking_bottleneck', 'shipping_bottleneck']

# B) 로봇 가용 용량 피처
print("  B) 로봇 가용 용량 피처...", flush=True)
combined['available_capacity'] = combined['robot_idle'] - (combined['robot_total'] * combined['low_battery_ratio'])
combined['charging_return_ratio'] = combined['robot_charging'] / (combined['robot_total'] + 1)
combined['robot_shortage'] = combined['order_inflow_15m'] / (combined['robot_idle'] + 1)
combined['effective_robot_ratio'] = (combined['robot_active'] - combined['robot_charging']) / (combined['robot_total'] + 1)
combined['robot_demand_balance'] = combined['robot_active'] / (combined['order_inflow_15m'] + 1)
domain_features += ['available_capacity', 'charging_return_ratio', 'robot_shortage',
                    'effective_robot_ratio', 'robot_demand_balance']

# C) 누적 피로도 피처
print("  C) 누적 피로도 피처...", flush=True)
combined['scenario_progress'] = combined['implicit_timeslot'] / 24.0
# battery_drain_rate: battery_mean_diff1 이미 존재하므로 재사용
combined['battery_drain_rate'] = combined['battery_mean_diff1']
# congestion_acceleration: congestion_score_diff1 이미 존재하므로 재사용
combined['congestion_acceleration'] = combined['congestion_score_diff1']
combined['late_scenario_flag'] = (combined['implicit_timeslot'] >= 19).astype(int)
combined['fatigue_index'] = combined['scenario_progress'] * (1 - combined['battery_mean'] / 100)
domain_features += ['scenario_progress', 'battery_drain_rate', 'congestion_acceleration',
                    'late_scenario_flag', 'fatigue_index']

# D) 주문 특성 x 창고 구조 궁합 피처
print("  D) 주문 특성 x 창고 구조 궁합 피처...", flush=True)
combined['complex_in_narrow'] = combined['avg_items_per_order'] / (combined['aisle_width_avg'] + 0.01)
combined['urgent_pack_pressure'] = combined['urgent_order_ratio'] * combined['order_inflow_15m'] / (combined['pack_station_count'] + 1)
combined['heavy_height_penalty'] = combined['heavy_item_ratio'] * combined['racking_height_avg_m']
combined['sku_per_intersection'] = combined['unique_sku_15m'] / (combined['intersection_count'] + 1)
combined['order_density_per_area'] = combined['order_inflow_15m'] / (combined['floor_area_sqm'] + 1) * 10000
domain_features += ['complex_in_narrow', 'urgent_pack_pressure', 'heavy_height_penalty',
                    'sku_per_intersection', 'order_density_per_area']

# E) 종합 위험도 지표
print("  E) 종합 위험도 지표...", flush=True)
combined['risk_score'] = (combined['low_battery_ratio'] * 0.3
                          + combined['congestion_score'] / 100 * 0.3
                          + combined['pack_utilization'] * 0.2
                          + combined['loading_dock_util'] * 0.2)
combined['capacity_stress'] = (combined['robot_utilization'] + combined['pack_utilization'] + combined['loading_dock_util']) / 3
domain_features += ['risk_score', 'capacity_stress']

print(f"\n  도메인 신규 피처 {len(domain_features)}개 추가 완료:", flush=True)
for f in domain_features:
    print(f"    - {f}", flush=True)

# ============================================================
# 원본 순서 복원 후 분리
# ============================================================
combined = combined.sort_values('_original_idx').reset_index(drop=True)
train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()

drop_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']
feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
print(f"\n총 피처 수: {len(feature_cols)}", flush=True)

X = train_fe[feature_cols]
y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y)
groups = train_fe['scenario_id']
X_test = test_fe[feature_cols]

# ID 순서 검증
assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID 순서 불일치!"
print("ID 순서 검증 통과!", flush=True)

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y_log, groups))

# ============================================================
# 3. LightGBM 단독 CV (도메인 피처 효과 확인, Phase 3B params)
# ============================================================
print("\n=== LightGBM 단독 CV (도메인 피처 효과 확인) ===", flush=True)

phase3b_params = {
    'objective': 'mae',
    'n_estimators': 3000,
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

def train_lgb_cv(X_use, X_test_use, y_log, y, folds, params):
    """LightGBM CV -> (oof, test_pred, cv_mae, models)"""
    oof = np.zeros(len(X_use))
    test_pred = np.zeros(len(X_test_use))
    models = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X_use.iloc[tr_idx], X_use.iloc[va_idx]
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred_log = m.predict(Xva)
        pred = np.clip(np.expm1(pred_log), 0, None)
        oof[va_idx] = pred
        test_pred += np.clip(np.expm1(m.predict(X_test_use)), 0, None) / len(folds)
        mae = mean_absolute_error(y.iloc[va_idx], pred)
        print(f"  Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)
        models.append(m)
    cv_mae = mean_absolute_error(y, oof)
    return oof, test_pred, cv_mae, models

lgb_domain_oof, _, lgb_domain_mae, _ = train_lgb_cv(X, X_test, y_log, y, folds, phase3b_params)
print(f"\n  도메인 피처 추가 전 (Phase 3B): 8.7908 -> 추가 후: {lgb_domain_mae:.4f}", flush=True)
print(f"  변화량: {lgb_domain_mae - 8.7908:+.4f}", flush=True)

# ============================================================
# 4. Optuna 재튜닝 (도메인 피처 포함, 50 trials)
# ============================================================
print("\n=== Optuna 하이퍼파라미터 재튜닝 (50 trials) ===", flush=True)

def objective(trial):
    params = {
        'objective': 'mae',
        'n_estimators': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    oof = np.zeros(len(X))
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred_log = m.predict(Xva)
        oof[va_idx] = np.clip(np.expm1(pred_log), 0, None)
    mae = mean_absolute_error(y, oof)
    return mae

def optuna_callback(study, trial):
    if (trial.number + 1) % 10 == 0:
        print(f"  Trial {trial.number+1}/50: best MAE={study.best_value:.4f}", flush=True)

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, callbacks=[optuna_callback])

best_params = study.best_params
best_optuna_mae = study.best_value
print(f"\n  Optuna 최적 MAE: {best_optuna_mae:.4f}", flush=True)
print(f"  최적 파라미터: {best_params}", flush=True)

# ============================================================
# 5. 타겟 변환 비교 (Optuna best params로)
# ============================================================
print("\n=== 타겟 변환 비교 ===", flush=True)

optuna_full_params = {
    'objective': 'mae',
    'n_estimators': 3000,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    **best_params,
}

transform_results = {}

# (a) log1p
print("\n  [1] log1p 변환...", flush=True)
y_log1p = np.log1p(y)
oof_log1p = np.zeros(len(X))
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y_log1p.iloc[tr_idx], y_log1p.iloc[va_idx]
    m = lgb.LGBMRegressor(**optuna_full_params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_log1p[va_idx] = np.clip(np.expm1(m.predict(Xva)), 0, None)
mae_log1p = mean_absolute_error(y, oof_log1p)
transform_results['log1p'] = mae_log1p
print(f"  log1p CV MAE: {mae_log1p:.4f}", flush=True)

# (b) sqrt
print("\n  [2] sqrt 변환...", flush=True)
y_sqrt = np.sqrt(y)
oof_sqrt = np.zeros(len(X))
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y_sqrt.iloc[tr_idx], y_sqrt.iloc[va_idx]
    m = lgb.LGBMRegressor(**optuna_full_params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred_sqrt = m.predict(Xva)
    oof_sqrt[va_idx] = np.clip(pred_sqrt ** 2, 0, None)
mae_sqrt = mean_absolute_error(y, oof_sqrt)
transform_results['sqrt'] = mae_sqrt
print(f"  sqrt CV MAE: {mae_sqrt:.4f}", flush=True)

# (c) 변환 없음 + objective='mae'
print("\n  [3] 변환 없음 + objective='mae'...", flush=True)
raw_params = optuna_full_params.copy()
oof_raw = np.zeros(len(X))
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    m = lgb.LGBMRegressor(**raw_params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_raw[va_idx] = np.clip(m.predict(Xva), 0, None)
mae_raw = mean_absolute_error(y, oof_raw)
transform_results['raw_mae'] = mae_raw
print(f"  raw+mae CV MAE: {mae_raw:.4f}", flush=True)

# (d) 변환 없음 + objective='huber' (alpha=10)
print("\n  [4] 변환 없음 + objective='huber' (alpha=10)...", flush=True)
huber_params = optuna_full_params.copy()
huber_params['objective'] = 'huber'
huber_params['huber_delta'] = 10
oof_huber = np.zeros(len(X))
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    m = lgb.LGBMRegressor(**huber_params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_huber[va_idx] = np.clip(m.predict(Xva), 0, None)
mae_huber = mean_absolute_error(y, oof_huber)
transform_results['raw_huber'] = mae_huber
print(f"  raw+huber CV MAE: {mae_huber:.4f}", flush=True)

# 최적 변환 선택
best_transform = min(transform_results, key=transform_results.get)
best_transform_mae = transform_results[best_transform]
print(f"\n  타겟 변환 비교 결과:", flush=True)
for name, mae in sorted(transform_results.items(), key=lambda x: x[1]):
    marker = " <-- best" if name == best_transform else ""
    print(f"    {name:12s}: {mae:.4f}{marker}", flush=True)

# ============================================================
# 6. 최종 앙상블
# ============================================================
print("\n=== 최종 앙상블 ===", flush=True)

# --- LightGBM (최적 변환 + Optuna best params) ---
print(f"  LightGBM 학습 (변환: {best_transform})...", flush=True)

if best_transform == 'log1p':
    y_train_transformed = np.log1p(y)
    inv_fn = lambda x: np.clip(np.expm1(x), 0, None)
    lgb_final_params = optuna_full_params.copy()
elif best_transform == 'sqrt':
    y_train_transformed = np.sqrt(y)
    inv_fn = lambda x: np.clip(x ** 2, 0, None)
    lgb_final_params = optuna_full_params.copy()
elif best_transform == 'raw_mae':
    y_train_transformed = y.copy()
    inv_fn = lambda x: np.clip(x, 0, None)
    lgb_final_params = optuna_full_params.copy()
elif best_transform == 'raw_huber':
    y_train_transformed = y.copy()
    inv_fn = lambda x: np.clip(x, 0, None)
    lgb_final_params = huber_params.copy()

lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))
lgb_maes = []
lgb_models = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr = y_train_transformed.iloc[tr_idx]
    yva = y_train_transformed.iloc[va_idx]
    m = lgb.LGBMRegressor(**lgb_final_params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = inv_fn(m.predict(Xva))
    lgb_oof[va_idx] = pred
    lgb_test += inv_fn(m.predict(X_test)) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    lgb_maes.append(mae)
    lgb_models.append(m)
    print(f"    LGB Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

lgb_cv = mean_absolute_error(y, lgb_oof)
print(f"  LightGBM CV MAE: {lgb_cv:.4f}", flush=True)

# --- CatBoost (같은 피처, log1p target, 기본 params) ---
print("  CatBoost 학습...", flush=True)
cat_oof = np.zeros(len(X))
cat_test = np.zeros(len(X_test))
cat_maes = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=6,
        loss_function='MAE', random_seed=42, verbose=0
    )
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100)
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    cat_oof[va_idx] = pred
    cat_test += np.clip(np.expm1(m.predict(X_test)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    cat_maes.append(mae)
    print(f"    Cat Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

cat_cv = mean_absolute_error(y, cat_oof)
print(f"  CatBoost CV MAE: {cat_cv:.4f}", flush=True)

# --- 앙상블 가중치 최적화 (2모델) ---
print("\n=== 앙상블 가중치 최적화 (LGB + Cat) ===", flush=True)

def ensemble_mae(weights):
    w = weights / weights.sum()
    blended = w[0] * lgb_oof + w[1] * cat_oof
    return mean_absolute_error(y, blended)

result = minimize(
    ensemble_mae,
    x0=np.array([0.5, 0.5]),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)

best_w = result.x / result.x.sum()
ensemble_oof = best_w[0] * lgb_oof + best_w[1] * cat_oof
ens_cv = mean_absolute_error(y, ensemble_oof)

print(f"  최적 가중치 - LightGBM: {best_w[0]:.4f}, CatBoost: {best_w[1]:.4f}", flush=True)
print(f"  앙상블 CV MAE: {ens_cv:.4f}", flush=True)

final_cv = ens_cv
final_test = best_w[0] * lgb_test + best_w[1] * cat_test

# ============================================================
# 7. 결과 비교 테이블
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 6 결과 ===", flush=True)
print("=" * 60, flush=True)

# 도메인 피처 중 중요도 Top 30 진입 수 계산
m_imp = lgb_models[0]
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': m_imp.feature_importances_
}).sort_values('importance', ascending=False)
top30_features = importance.head(30)['feature'].tolist()
domain_in_top30 = [f for f in domain_features if f in top30_features]

print(f"도메인 피처 추가 후 (기존 params):  LGB CV MAE {lgb_domain_mae:.4f} (vs Phase3B 8.7908)", flush=True)
print(f"Optuna 재튜닝 후:                   LGB CV MAE {best_optuna_mae:.4f}", flush=True)
print(f"최적 타겟 변환:                     [{best_transform}] CV MAE {best_transform_mae:.4f}", flush=True)
print(f"LightGBM 최종:                      CV MAE {lgb_cv:.4f}", flush=True)
print(f"CatBoost:                           CV MAE {cat_cv:.4f}", flush=True)
print(f"앙상블:                             CV MAE {ens_cv:.4f}", flush=True)
print(f"Phase 3B 대비 개선:                 {ens_cv - 8.7908:+.4f}", flush=True)
print(f"총 피처 수:                         {len(feature_cols)}개", flush=True)
print(f"도메인 피처 중 중요도 Top 30 진입:  {len(domain_in_top30)}개", flush=True)
if domain_in_top30:
    for f in domain_in_top30:
        rank = top30_features.index(f) + 1
        print(f"  #{rank:2d} {f}", flush=True)

# ============================================================
# 8. 제출 파일 생성
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
final_test = np.clip(final_test, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase6.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase6.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 9. 피처 중요도 시각화
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)

top30 = importance.head(30).sort_values('importance')
ts_suffixes = ['_lag', '_roll', '_diff1', '_cumsum']

def get_color(f):
    if f in domain_features:
        return 'darkgreen'
    elif f in ['orders_per_packstation', 'pack_dock_pressure', 'dock_wait_pressure',
               'shift_load_pressure', 'battery_congestion', 'storage_density_congestion',
               'battery_trip_pressure', 'demand_density']:
        return 'forestgreen'
    elif any(s in f for s in ts_suffixes):
        return 'coral'
    else:
        return 'steelblue'

colors = [get_color(f) for f in top30['feature']]

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color=colors)
ax.set_title('피처 중요도 Top 30 (Phase 6 - 도메인 피처 + Optuna)')
ax.set_xlabel('중요도')
ax.legend(handles=[
    Patch(color='coral', label='시계열 피처'),
    Patch(color='steelblue', label='기존 피처'),
    Patch(color='forestgreen', label='Phase 3B 인터랙션'),
    Patch(color='darkgreen', label='도메인 신규 피처')
])
plt.tight_layout()
plt.savefig('output/feature_importance_phase6.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase6.png 저장 완료", flush=True)

print(f"\n도메인 피처 중 중요도 Top 30 진입: {len(domain_in_top30)}개", flush=True)

print("\n=== Phase 6 완료 ===", flush=True)
