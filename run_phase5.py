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
import warnings, sys, time
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Optuna best params from Phase 3B
OPTUNA_PARAMS = dict(
    objective='mae', n_estimators=3000,
    learning_rate=0.0129, num_leaves=185, max_depth=9,
    min_child_samples=80, reg_alpha=0.0574, reg_lambda=0.0042,
    feature_fraction=0.6005, bagging_fraction=0.7663, bagging_freq=1,
    random_state=42, n_jobs=-1, verbose=-1
)

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

print(f"  시계열 피처 생성 완료", flush=True)

# --- layout 조인 + 기존 피처 ---
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

# --- 인터랙션 피처 (Phase 3B) ---
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

print(f"  기존 피처 생성 완료", flush=True)

# ============================================================
# 2. 대량 GroupBy 피처 생성
# ============================================================
print("\n=== 대량 GroupBy 피처 생성 ===", flush=True)

group_keys = ['layout_id', 'layout_type', 'implicit_timeslot', 'day_of_week']
agg_targets = [
    'order_inflow_15m', 'unique_sku_15m', 'robot_active', 'robot_idle', 'robot_charging',
    'battery_mean', 'low_battery_ratio', 'congestion_score', 'max_zone_density', 'pack_utilization',
    'robot_utilization', 'loading_dock_util', 'avg_trip_distance', 'charge_queue_length', 'fault_count_15m'
]
stats = ['mean', 'std', 'max', 'min', 'median']

groupby_features = []
total_combos = len(group_keys) * len(agg_targets) * len(stats)
created = 0
t0 = time.time()

for col1 in group_keys:
    for col2 in agg_targets:
        agg_dict = combined.groupby(col1)[col2].agg(stats)
        for stat in stats:
            new_col = f'{col2}_by_{col1}_{stat}'
            mapping = agg_dict[stat]
            combined[new_col] = combined[col1].map(mapping)
            groupby_features.append(new_col)
            created += 1
    print(f"  {col1} 완료 ({created}/{total_combos} 피처, {time.time()-t0:.1f}s)", flush=True)

print(f"  GroupBy 피처 생성 완료: {len(groupby_features)}개 ({time.time()-t0:.1f}s)", flush=True)

# ============================================================
# 3. 타겟 Lag 피처 (train만 실제값, test는 재귀 예측에서 채움)
# ============================================================
print("\n=== 타겟 Lag 피처 생성 (train) ===", flush=True)

target_col = 'avg_delay_minutes_next_30m'

# train에서 실제 타겟값으로 lag 생성
combined['target_lag1'] = combined.groupby('scenario_id')[target_col].shift(1)
combined['target_lag2'] = combined.groupby('scenario_id')[target_col].shift(2)
combined['target_lag3'] = combined.groupby('scenario_id')[target_col].shift(3)

# rolling mean (lag 기반)
tl1 = combined['target_lag1']
tl2 = combined['target_lag2']
tl3 = combined['target_lag3']
combined['target_roll3_mean'] = pd.concat([tl1, tl2, tl3], axis=1).mean(axis=1)

# cummax, cummean (shift해서 현재 슬롯 미포함)
combined['target_cummax'] = combined.groupby('scenario_id')[target_col].apply(
    lambda s: s.shift(1).expanding().max()
).reset_index(level=0, drop=True)
combined['target_cummean'] = combined.groupby('scenario_id')[target_col].apply(
    lambda s: s.shift(1).expanding().mean()
).reset_index(level=0, drop=True)

target_lag_features = ['target_lag1', 'target_lag2', 'target_lag3',
                       'target_roll3_mean', 'target_cummax', 'target_cummean']
print(f"  타겟 Lag 피처 6개 생성 완료", flush=True)

# ============================================================
# 원본 순서 복원 후 분리
# ============================================================
combined = combined.sort_values('_original_idx').reset_index(drop=True)
train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()

drop_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type',
             target_col, '_is_train', '_original_idx']

# 모델 A용 피처: 타겟 lag 제외
feature_cols_no_tlag = [c for c in train_fe.columns
                        if c not in drop_cols and c not in target_lag_features
                        and c in test_fe.columns]

# 모델 B용 피처: 타겟 lag 포함
feature_cols_with_tlag = feature_cols_no_tlag + target_lag_features

print(f"\n전체 피처 수 (타겟lag 제외): {len(feature_cols_no_tlag)}", flush=True)
print(f"전체 피처 수 (타겟lag 포함): {len(feature_cols_with_tlag)}", flush=True)

y = train_fe[target_col]
y_log = np.log1p(y)
groups = train_fe['scenario_id']

# ID 순서 검증
assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID 순서 불일치!"
print("ID 순서 검증 통과!", flush=True)

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(train_fe, y_log, groups))

# ============================================================
# 4. 피처 선별 — 중요도 기반 (3-Fold 빠른 학습)
# ============================================================
print("\n=== 피처 선별: 중요도 기반 (3-Fold) ===", flush=True)
gkf3 = GroupKFold(n_splits=3)
folds3 = list(gkf3.split(train_fe, y_log, groups))

# 모델 A용 피처 선별
print("  모델 A 피처 중요도 계산 중...", flush=True)
X_all_no_tlag = train_fe[feature_cols_no_tlag]
imp_sum = np.zeros(len(feature_cols_no_tlag))

for fold_i, (tr_idx, va_idx) in enumerate(folds3):
    m = lgb.LGBMRegressor(**{**OPTUNA_PARAMS, 'n_estimators': 1000})
    m.fit(X_all_no_tlag.iloc[tr_idx], y_log.iloc[tr_idx],
          eval_set=[(X_all_no_tlag.iloc[va_idx], y_log.iloc[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    imp_sum += m.feature_importances_
    print(f"    Fold {fold_i+1}/3 완료", flush=True)

importance_a = pd.DataFrame({
    'feature': feature_cols_no_tlag,
    'importance': imp_sum / 3
}).sort_values('importance', ascending=False)

# 상위 300개 선택
n_select = min(300, len(feature_cols_no_tlag))
selected_features_a = importance_a.head(n_select)['feature'].tolist()
removed_features_a = importance_a.tail(len(feature_cols_no_tlag) - n_select)['feature'].tolist()
print(f"  모델 A: {len(feature_cols_no_tlag)} → {len(selected_features_a)}개 피처 선택", flush=True)
print(f"  제거된 피처 수: {len(removed_features_a)}개", flush=True)
if len(removed_features_a) > 0:
    print(f"  제거 예시: {removed_features_a[:10]}", flush=True)

# 모델 B용 피처 선별 (타겟 lag 포함)
print("\n  모델 B 피처 중요도 계산 중...", flush=True)
X_all_with_tlag = train_fe[feature_cols_with_tlag]
imp_sum_b = np.zeros(len(feature_cols_with_tlag))

for fold_i, (tr_idx, va_idx) in enumerate(folds3):
    m = lgb.LGBMRegressor(**{**OPTUNA_PARAMS, 'n_estimators': 1000})
    m.fit(X_all_with_tlag.iloc[tr_idx], y_log.iloc[tr_idx],
          eval_set=[(X_all_with_tlag.iloc[va_idx], y_log.iloc[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    imp_sum_b += m.feature_importances_
    print(f"    Fold {fold_i+1}/3 완료", flush=True)

importance_b = pd.DataFrame({
    'feature': feature_cols_with_tlag,
    'importance': imp_sum_b / 3
}).sort_values('importance', ascending=False)

n_select_b = min(300, len(feature_cols_with_tlag))
selected_features_b = importance_b.head(n_select_b)['feature'].tolist()
# 타겟 lag 피처는 반드시 포함
for tf in target_lag_features:
    if tf not in selected_features_b:
        selected_features_b.append(tf)
print(f"  모델 B: {len(feature_cols_with_tlag)} → {len(selected_features_b)}개 피처 선택", flush=True)

# ============================================================
# 5. 모델 A: 일반 LightGBM (Optuna params, 선별 피처)
# ============================================================
print("\n=== 모델 A: LightGBM (GroupBy 피처, 타겟lag 제외) ===", flush=True)
X_a = train_fe[selected_features_a]
X_test_a = test_fe[selected_features_a]

lgb_a_oof = np.zeros(len(X_a))
lgb_a_test = np.zeros(len(X_test_a))
lgb_a_maes = []
lgb_a_models = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X_a.iloc[tr_idx], X_a.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = lgb.LGBMRegressor(**OPTUNA_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    lgb_a_oof[va_idx] = pred
    lgb_a_test += np.clip(np.expm1(m.predict(X_test_a)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    lgb_a_maes.append(mae)
    lgb_a_models.append(m)
    print(f"  모델A Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

lgb_a_cv = mean_absolute_error(y, lgb_a_oof)
print(f"  모델 A CV MAE: {lgb_a_cv:.4f}", flush=True)

# ============================================================
# 6. 모델 B: 타겟 Lag 포함 LightGBM (CV + 재귀 예측)
# ============================================================
print("\n=== 모델 B: LightGBM (타겟 Lag 포함) ===", flush=True)

# --- CV (train: 실제 타겟값 사용, 낙관적) ---
X_b = train_fe[selected_features_b]
lgb_b_oof = np.zeros(len(X_b))
lgb_b_maes = []
lgb_b_models = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X_b.iloc[tr_idx], X_b.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = lgb.LGBMRegressor(**OPTUNA_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    lgb_b_oof[va_idx] = pred
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    lgb_b_maes.append(mae)
    lgb_b_models.append(m)
    print(f"  모델B Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

lgb_b_cv = mean_absolute_error(y, lgb_b_oof)
print(f"  모델 B CV MAE: {lgb_b_cv:.4f} (train 실제값 기준, 낙관적)", flush=True)

# --- 모델 B: 전체 train으로 재학습 (test 재귀 예측용) ---
print("\n  모델 B 전체 train 재학습 (재귀 예측용)...", flush=True)
m_b_full = lgb.LGBMRegressor(**OPTUNA_PARAMS)
m_b_full.fit(X_b, y_log)
print(f"  전체 학습 완료", flush=True)

# --- Test 재귀 예측 ---
print("\n=== 모델 B: Test 재귀 예측 ===", flush=True)

# test_fe에 타겟 lag 컬럼을 NaN으로 초기화
test_recursive = test_fe.copy()
for tf in target_lag_features:
    test_recursive[tf] = np.nan

# 시나리오별, 슬롯 순서대로 재귀 예측
test_scenarios = test_recursive.groupby('scenario_id')
scenario_ids = test_recursive['scenario_id'].unique()
n_scenarios = len(scenario_ids)

lgb_b_test = np.full(len(test_recursive), np.nan)
idx_to_pos = {idx: pos for pos, idx in enumerate(test_recursive.index)}
t0 = time.time()

for s_idx, sid in enumerate(scenario_ids):
    sc_mask = test_recursive['scenario_id'] == sid
    sc_indices = test_recursive.index[sc_mask]
    sc_data = test_recursive.loc[sc_indices].sort_values('implicit_timeslot')

    predictions = []  # 이 시나리오 내 슬롯별 예측값
    for row_i, (idx, row) in enumerate(sc_data.iterrows()):
        # 현재 슬롯의 target lag 피처 채우기
        if row_i >= 1:
            test_recursive.at[idx, 'target_lag1'] = predictions[-1]
        if row_i >= 2:
            test_recursive.at[idx, 'target_lag2'] = predictions[-2]
        if row_i >= 3:
            test_recursive.at[idx, 'target_lag3'] = predictions[-3]

        # target_roll3_mean
        recent = predictions[-3:] if len(predictions) >= 1 else []
        if len(recent) > 0:
            test_recursive.at[idx, 'target_roll3_mean'] = np.mean(recent[-3:])

        # target_cummax, target_cummean
        if len(predictions) > 0:
            test_recursive.at[idx, 'target_cummax'] = np.max(predictions)
            test_recursive.at[idx, 'target_cummean'] = np.mean(predictions)

        # 예측
        X_row = test_recursive.loc[[idx], selected_features_b]
        pred_log = m_b_full.predict(X_row)[0]
        pred = max(np.expm1(pred_log), 0)
        predictions.append(pred)
        lgb_b_test[idx_to_pos[idx]] = pred

    if (s_idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  재귀 예측 진행: {s_idx+1}/{n_scenarios} 시나리오 ({elapsed:.1f}s)", flush=True)

print(f"  재귀 예측 완료: {n_scenarios} 시나리오 ({time.time()-t0:.1f}s)", flush=True)
assert not np.isnan(lgb_b_test).any(), "재귀 예측에 NaN 존재!"

# ============================================================
# 7. CatBoost (모델 A와 같은 피처)
# ============================================================
print("\n=== CatBoost 학습 (모델 A 피처) ===", flush=True)
cat_oof = np.zeros(len(X_a))
cat_test = np.zeros(len(X_test_a))
cat_maes = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X_a.iloc[tr_idx], X_a.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=6,
        loss_function='MAE', random_seed=42, verbose=0
    )
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100)
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    cat_oof[va_idx] = pred
    cat_test += np.clip(np.expm1(m.predict(X_test_a)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    cat_maes.append(mae)
    print(f"  Cat Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

cat_cv = mean_absolute_error(y, cat_oof)
print(f"  CatBoost CV MAE: {cat_cv:.4f}", flush=True)

# ============================================================
# 8. 앙상블 가중치 최적화
# ============================================================
print("\n=== 앙상블 가중치 최적화 ===", flush=True)

def ensemble_mae(weights):
    w = weights / weights.sum()
    blended = w[0] * lgb_a_oof + w[1] * lgb_b_oof + w[2] * cat_oof
    return mean_absolute_error(y, blended)

result = minimize(
    ensemble_mae,
    x0=np.array([1/3, 1/3, 1/3]),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)

best_w = result.x / result.x.sum()
ensemble_oof = best_w[0] * lgb_a_oof + best_w[1] * lgb_b_oof + best_w[2] * cat_oof
ens_cv = mean_absolute_error(y, ensemble_oof)

print(f"최적 가중치 - 모델A: {best_w[0]:.4f}, 모델B: {best_w[1]:.4f}, CatBoost: {best_w[2]:.4f}", flush=True)
print(f"앙상블 CV MAE: {ens_cv:.4f}", flush=True)

# ============================================================
# 9. 결과 비교 테이블
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 5 결과 ===", flush=True)
print("=" * 60, flush=True)
print(f"생성된 GroupBy 피처: {len(groupby_features)}개", flush=True)
print(f"선별 후 피처 수 (모델A): {len(selected_features_a)}개", flush=True)
print(f"선별 후 피처 수 (모델B): {len(selected_features_b)}개", flush=True)
print(f"모델 A (GroupBy 피처):     CV MAE {lgb_a_cv:.4f}", flush=True)
print(f"모델 B (타겟 Lag 포함):    CV MAE {lgb_b_cv:.4f} (train 실제값 기준, 낙관적)", flush=True)
print(f"CatBoost:                  CV MAE {cat_cv:.4f}", flush=True)
print(f"최종 앙상블:               CV MAE {ens_cv:.4f}", flush=True)
print(f"Phase 3B 앙상블:           8.786", flush=True)
print(f"Phase 3B 대비 개선:        {ens_cv - 8.786:+.4f}", flush=True)

# ============================================================
# 10. 제출 파일 생성
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
test_blended = best_w[0] * lgb_a_test + best_w[1] * lgb_b_test + best_w[2] * cat_test
test_blended = np.clip(test_blended, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = test_blended
submission.to_csv('output/submission_phase5.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase5.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 11. 피처 중요도 시각화
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)
importance_final = pd.DataFrame({
    'feature': selected_features_a,
    'importance': np.mean([m.feature_importances_ for m in lgb_a_models], axis=0)
}).sort_values('importance', ascending=False)

top30 = importance_final.head(30).sort_values('importance')
ts_suffixes = ['_lag', '_roll', '_diff1', '_cumsum']

def get_color(f):
    if '_by_' in f:
        return 'forestgreen'
    elif any(s in f for s in ts_suffixes):
        return 'coral'
    else:
        return 'steelblue'

colors = [get_color(f) for f in top30['feature']]

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color=colors)
ax.set_title('피처 중요도 Top 30 (Phase 5 — 모델 A)')
ax.set_xlabel('중요도')
ax.legend(handles=[
    Patch(color='coral', label='시계열 피처'),
    Patch(color='steelblue', label='기존 피처'),
    Patch(color='forestgreen', label='GroupBy 피처')
])
plt.tight_layout()
plt.savefig('output/feature_importance_phase5.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase5.png 저장 완료", flush=True)

print("\n=== Phase 5 완료 ===", flush=True)
