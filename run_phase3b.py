import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from matplotlib.patches import Patch
import optuna
import warnings, sys
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 데이터 준비 (Phase 2 로직 재사용)
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

# --- Phase 3B 신규: 잔차 분석 기반 인터랙션 피처 ---
print("=== 잔차 분석 기반 신규 피처 추가 ===", flush=True)
combined['orders_per_packstation'] = combined['order_inflow_15m'] / combined['pack_station_count'].replace(0, np.nan)
combined['pack_dock_pressure'] = combined['pack_utilization'] * combined['loading_dock_util']
combined['dock_wait_pressure'] = combined['outbound_truck_wait_min'] * combined['loading_dock_util']
combined['shift_load_pressure'] = combined['prev_shift_volume'] * combined['order_inflow_15m']
combined['battery_congestion'] = combined['low_battery_ratio'] * combined['congestion_score']
combined['storage_density_congestion'] = combined['storage_density_pct'] * combined['congestion_score']
combined['battery_trip_pressure'] = combined['low_battery_ratio'] * combined['avg_trip_distance']
combined['demand_density'] = combined['order_inflow_15m'] * combined['max_zone_density']
print("  신규 인터랙션 피처 8개 추가 완료", flush=True)

# --- 결측 지시자 ---
train_part = combined[combined['_is_train'] == 1]
missing_counts = train_part.isnull().sum().sort_values(ascending=False)
top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith('_')]
for col in top10_missing:
    if col in combined.columns:
        combined[f'{col}_missing'] = combined[col].isnull().astype(int)

# 원본 순서 복원 후 분리
combined = combined.sort_values('_original_idx').reset_index(drop=True)
train_fe = combined[combined['_is_train'] == 1].copy()
test_fe = combined[combined['_is_train'] == 0].copy()

drop_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type', 'avg_delay_minutes_next_30m', '_is_train', '_original_idx']
feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
print(f"총 피처 수: {len(feature_cols)}", flush=True)

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
# 2. 신규 피처 효과 확인 (LightGBM 단독)
# ============================================================
print("\n=== 신규 피처 효과 확인 (LightGBM) ===", flush=True)

def train_lgb_cv(X_use, X_test_use, y_log, y, folds, params=None):
    """LightGBM CV → (oof, test_pred, cv_mae, models)"""
    base_params = dict(
        objective='mae', n_estimators=2000, learning_rate=0.03,
        num_leaves=63, random_state=42, n_jobs=-1, verbose=-1
    )
    if params:
        base_params.update(params)
    oof = np.zeros(len(X_use))
    test_pred = np.zeros(len(X_test_use))
    models = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X_use.iloc[tr_idx], X_use.iloc[va_idx]
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
        m = lgb.LGBMRegressor(**base_params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred_log = m.predict(Xva)
        pred = np.clip(np.expm1(pred_log), 0, None)
        oof[va_idx] = pred
        test_pred += np.clip(np.expm1(m.predict(X_test_use)), 0, None) / len(folds)
        models.append(m)
    cv_mae = mean_absolute_error(y, oof)
    return oof, test_pred, cv_mae, models

lgb_oof_base, _, lgb_base_mae, _ = train_lgb_cv(X, X_test, y_log, y, folds)
print(f"  신규 피처 포함 LGB CV MAE: {lgb_base_mae:.4f} (vs Phase2 8.8508)", flush=True)

# ============================================================
# 3. 듀얼 모델 전략 — 고지연 특화 모델
# ============================================================
print("\n=== 듀얼 모델 전략 ===", flush=True)

thresholds = [10, 15, 20, 25, 30]
best_threshold = None
best_dual_mae = float('inf')
best_dual_oof = None

for thr in thresholds:
    dual_oof = np.zeros(len(X))
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr_log, yva_log = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        # 모델 A: 전체 데이터
        mA = lgb.LGBMRegressor(
            objective='mae', n_estimators=2000, learning_rate=0.03,
            num_leaves=63, random_state=42, n_jobs=-1, verbose=-1
        )
        mA.fit(Xtr, ytr_log, eval_set=[(Xva, yva_log)],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        predA = np.clip(np.expm1(mA.predict(Xva)), 0, None)

        # 모델 B: 고지연 데이터만
        high_mask = y.iloc[tr_idx] >= thr
        if high_mask.sum() < 50:
            dual_oof[va_idx] = predA
            continue
        Xtr_high = Xtr[high_mask.values]
        ytr_high_log = ytr_log[high_mask.values]

        mB = lgb.LGBMRegressor(
            objective='mae', n_estimators=2000, learning_rate=0.03,
            num_leaves=63, random_state=42, n_jobs=-1, verbose=-1
        )
        mB.fit(Xtr_high, ytr_high_log, eval_set=[(Xva, yva_log)],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        predB = np.clip(np.expm1(mB.predict(Xva)), 0, None)

        # 결합: A 예측이 임계값 이상이면 B 사용
        combined_pred = np.where(predA >= thr, predB, predA)
        dual_oof[va_idx] = combined_pred

    dual_mae = mean_absolute_error(y, dual_oof)
    print(f"  임계값 {thr:2d}분: 듀얼 모델 CV MAE {dual_mae:.4f}", flush=True)
    if dual_mae < best_dual_mae:
        best_dual_mae = dual_mae
        best_threshold = thr
        best_dual_oof = dual_oof.copy()

print(f"  → 최적 임계값: {best_threshold}분, 듀얼 모델 CV MAE: {best_dual_mae:.4f}", flush=True)
print(f"  → 단일 모델 대비 변화: {best_dual_mae - lgb_base_mae:+.4f}", flush=True)

# ============================================================
# 4. Optuna 튜닝 (모델 A, 50 trials)
# ============================================================
print("\n=== Optuna 하이퍼파라미터 튜닝 (50 trials) ===", flush=True)

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

study = optuna.create_study(direction='minimize', seed=42)
study.optimize(objective, n_trials=50, callbacks=[optuna_callback])

best_params = study.best_params
best_optuna_mae = study.best_value
print(f"\n  Optuna 최적 MAE: {best_optuna_mae:.4f}", flush=True)
print(f"  최적 파라미터: {best_params}", flush=True)

# ============================================================
# 5. 최종 앙상블 — Optuna LGB + CatBoost + XGBoost
# ============================================================
print("\n=== 최종 앙상블 ===", flush=True)

# --- LightGBM (Optuna best) ---
print("  LightGBM (Optuna best) 학습...", flush=True)
optuna_lgb_params = {
    'objective': 'mae', 'n_estimators': 3000,
    'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    **best_params
}
lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))
lgb_maes = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = lgb.LGBMRegressor(**optuna_lgb_params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    lgb_oof[va_idx] = pred
    lgb_test += np.clip(np.expm1(m.predict(X_test)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    lgb_maes.append(mae)
    print(f"    LGB Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

lgb_cv = mean_absolute_error(y, lgb_oof)
print(f"  LightGBM CV MAE: {lgb_cv:.4f}", flush=True)

# --- CatBoost ---
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

# --- XGBoost ---
print("  XGBoost 학습...", flush=True)
xgb_oof = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))
xgb_maes = []

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        eval_metric='mae', tree_method='hist', random_state=42,
        verbosity=0, early_stopping_rounds=100
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    xgb_oof[va_idx] = pred
    xgb_test += np.clip(np.expm1(m.predict(X_test)), 0, None) / 5
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    xgb_maes.append(mae)
    print(f"    XGB Fold {fold_i+1} MAE: {mae:.4f} (iter: {m.best_iteration})", flush=True)

xgb_cv = mean_absolute_error(y, xgb_oof)
print(f"  XGBoost CV MAE: {xgb_cv:.4f}", flush=True)

# --- 앙상블 가중치 최적화 ---
print("\n=== 앙상블 가중치 최적화 ===", flush=True)

def ensemble_mae(weights):
    w = weights / weights.sum()
    blended = w[0] * lgb_oof + w[1] * cat_oof + w[2] * xgb_oof
    return mean_absolute_error(y, blended)

result = minimize(
    ensemble_mae,
    x0=np.array([1/3, 1/3, 1/3]),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)

best_w = result.x / result.x.sum()
ensemble_oof = best_w[0] * lgb_oof + best_w[1] * cat_oof + best_w[2] * xgb_oof
ens_cv = mean_absolute_error(y, ensemble_oof)

print(f"최적 가중치 - LightGBM: {best_w[0]:.4f}, CatBoost: {best_w[1]:.4f}, XGBoost: {best_w[2]:.4f}", flush=True)
print(f"앙상블 CV MAE: {ens_cv:.4f}", flush=True)

# --- 듀얼 모델 적용 여부 결정 ---
print("\n=== 듀얼 모델 적용 여부 결정 ===", flush=True)
print(f"  단일 앙상블 CV MAE: {ens_cv:.4f}", flush=True)
print(f"  듀얼 모델 (LGB 단독) CV MAE: {best_dual_mae:.4f}", flush=True)

# 듀얼 모델은 LGB 단독 기준이므로, 앙상블과 공정 비교를 위해
# 앙상블 OOF에 듀얼 전략을 적용
if best_dual_mae < lgb_base_mae:
    print(f"  → 듀얼 모델이 LGB 단독 대비 개선됨. 앙상블에도 적용 시도...", flush=True)
    # 앙상블 OOF 기반으로 듀얼 전략 적용
    # 고지연 특화 모델을 앙상블 OOF에 대해서도 학습
    dual_ens_oof = np.zeros(len(X))
    dual_ens_test = np.zeros(len(X_test))

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr_log = y_log.iloc[tr_idx]
        yva_log = y_log.iloc[va_idx]

        # 고지연 모델 B
        high_mask = y.iloc[tr_idx] >= best_threshold
        Xtr_high = Xtr[high_mask.values]
        ytr_high_log = ytr_log[high_mask.values]

        mB = lgb.LGBMRegressor(**optuna_lgb_params)
        mB.fit(Xtr_high, ytr_high_log, eval_set=[(Xva, yva_log)],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        predB = np.clip(np.expm1(mB.predict(Xva)), 0, None)
        predB_test = np.clip(np.expm1(mB.predict(X_test)), 0, None)

        # 앙상블 예측이 임계값 이상이면 B 사용
        ens_va = best_w[0] * lgb_oof[va_idx] + best_w[1] * cat_oof[va_idx] + best_w[2] * xgb_oof[va_idx]
        dual_pred = np.where(ens_va >= best_threshold, predB, ens_va)
        dual_ens_oof[va_idx] = dual_pred

        ens_test_fold = best_w[0] * lgb_test + best_w[1] * cat_test + best_w[2] * xgb_test
        dual_ens_test += np.where(ens_test_fold >= best_threshold, predB_test, ens_test_fold) / 5

    dual_ens_mae = mean_absolute_error(y, dual_ens_oof)
    print(f"  듀얼 앙상블 CV MAE: {dual_ens_mae:.4f}", flush=True)

    if dual_ens_mae < ens_cv:
        print(f"  → 듀얼 앙상블 채택 (개선: {ens_cv - dual_ens_mae:+.4f})", flush=True)
        final_cv = dual_ens_mae
        final_test = dual_ens_test
        use_dual = True
    else:
        print(f"  → 단일 앙상블 유지 (듀얼이 더 나쁨: {dual_ens_mae - ens_cv:+.4f})", flush=True)
        final_cv = ens_cv
        final_test = best_w[0] * lgb_test + best_w[1] * cat_test + best_w[2] * xgb_test
        use_dual = False
else:
    print(f"  → 듀얼 모델이 개선 없음. 단일 앙상블 유지.", flush=True)
    final_cv = ens_cv
    final_test = best_w[0] * lgb_test + best_w[1] * cat_test + best_w[2] * xgb_test
    use_dual = False

# ============================================================
# 6. 결과 비교 테이블
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 3B 결과 ===", flush=True)
print("=" * 60, flush=True)
print(f"신규 피처 추가 효과:     LGB CV MAE {lgb_base_mae:.4f} (vs Phase2 8.8508)", flush=True)
print(f"듀얼 모델 (임계값={best_threshold}분): CV MAE {best_dual_mae:.4f}", flush=True)
print(f"Optuna 최적 LGB:        CV MAE {best_optuna_mae:.4f}", flush=True)
print(f"Optuna best params:     {best_params}", flush=True)
print(f"최종 LightGBM CV MAE:   {lgb_cv:.4f}", flush=True)
print(f"최종 CatBoost CV MAE:   {cat_cv:.4f}", flush=True)
print(f"최종 XGBoost CV MAE:    {xgb_cv:.4f}", flush=True)
print(f"앙상블 CV MAE:          {ens_cv:.4f}", flush=True)
print(f"듀얼 적용 여부:         {'Yes' if use_dual else 'No'}", flush=True)
print(f"최종 CV MAE:            {final_cv:.4f}", flush=True)
print(f"Phase 2 앙상블:         8.8253", flush=True)
print(f"개선폭:                 {final_cv - 8.8253:+.4f}", flush=True)

# ============================================================
# 7. 제출 파일 생성
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
final_test = np.clip(final_test, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = final_test
submission.to_csv('output/submission_phase3b.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase3b.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 8. 피처 중요도 시각화
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.mean([lgb.LGBMRegressor(**optuna_lgb_params).fit(
        X.iloc[folds[0][0]], y_log.iloc[folds[0][0]]).feature_importances_
        for _ in range(1)], axis=0)
})
# 단순화: fold 0 모델의 중요도 사용
m_imp = lgb.LGBMRegressor(**optuna_lgb_params)
m_imp.fit(X.iloc[folds[0][0]], y_log.iloc[folds[0][0]],
          eval_set=[(X.iloc[folds[0][1]], y_log.iloc[folds[0][1]])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': m_imp.feature_importances_
}).sort_values('importance', ascending=False)

top30 = importance.head(30).sort_values('importance')
ts_suffixes = ['_lag', '_roll', '_diff1', '_cumsum']
new_features = ['orders_per_packstation', 'pack_dock_pressure', 'dock_wait_pressure',
                'shift_load_pressure', 'battery_congestion', 'storage_density_congestion',
                'battery_trip_pressure', 'demand_density']

def get_color(f):
    if f in new_features:
        return 'forestgreen'
    elif any(s in f for s in ts_suffixes):
        return 'coral'
    else:
        return 'steelblue'

colors = [get_color(f) for f in top30['feature']]

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color=colors)
ax.set_title('피처 중요도 Top 30 (Phase 3B — Optuna LGB)')
ax.set_xlabel('중요도')
ax.legend(handles=[
    Patch(color='coral', label='시계열 피처'),
    Patch(color='steelblue', label='기존 피처'),
    Patch(color='forestgreen', label='신규 인터랙션 피처')
])
plt.tight_layout()
plt.savefig('output/feature_importance_phase3b.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase3b.png 저장 완료", flush=True)

print("\n=== Phase 3B 완료 ===", flush=True)
