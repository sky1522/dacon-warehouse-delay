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
import warnings, sys
warnings.filterwarnings('ignore')

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

# --- Phase 3A 신규: layout 메타데이터 파생 피처 ---
print("=== layout 파생 피처 추가 ===", flush=True)
combined['robot_per_packstation'] = combined['robot_total'] / combined['pack_station_count'].replace(0, np.nan)
combined['charger_density'] = combined['charger_count'] / combined['floor_area_sqm'].replace(0, np.nan)
combined['intersection_density'] = combined['intersection_count'] / combined['floor_area_sqm'].replace(0, np.nan)
combined['robot_compactness'] = combined['robot_total'] * combined['layout_compactness']
combined['dispersion_robot'] = combined['zone_dispersion'] * combined['robot_total']
print("  layout 파생 피처 5개 추가 완료", flush=True)

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
feature_cols_all = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
print(f"전체 피처 수 (layout 피처 추가 후): {len(feature_cols_all)}", flush=True)

# layout_id가 피처에 포함되지 않았는지 확인
assert 'layout_id' not in feature_cols_all, "layout_id가 피처에 포함되어 있음!"
print("layout_id 미포함 확인 완료", flush=True)

X_all = train_fe[feature_cols_all]
y = train_fe['avg_delay_minutes_next_30m']
y_log = np.log1p(y)
groups_scenario = train_fe['scenario_id']
groups_layout = train_fe['layout_id']
X_test_all = test_fe[feature_cols_all]

# ID 순서 검증
assert (test_fe['ID'].values == sample_sub['ID'].values).all(), "ID 순서 불일치!"
print("ID 순서 검증 통과!", flush=True)

# ============================================================
# 실험 0: Phase 2 재현 (기준선) — LightGBM 단독
# ============================================================
print("\n=== 실험 0: Phase 2 기준선 (LightGBM, 전체 피처, 기존 하이퍼파라미터) ===", flush=True)
gkf_scenario = GroupKFold(n_splits=5)
folds_scenario = list(gkf_scenario.split(X_all, y_log, groups_scenario))

def train_lgb_cv(X, y_log, y, folds, feature_cols, params_override=None):
    """LightGBM CV 학습 후 (oof_pred, test_pred, cv_mae, models) 반환"""
    base_params = dict(
        objective='mae', n_estimators=2000, learning_rate=0.03,
        num_leaves=63, random_state=42, n_jobs=-1, verbose=-1
    )
    if params_override:
        base_params.update(params_override)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test_all))
    models = []
    X_use = X[feature_cols] if isinstance(X, pd.DataFrame) else X
    X_test_use = X_test_all[feature_cols]

    for fold, (tr_idx, va_idx) in enumerate(folds):
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

_, _, baseline_lgb_mae, baseline_lgb_models = train_lgb_cv(
    X_all, y_log, y, folds_scenario, feature_cols_all
)
print(f"  기준선 LGB CV MAE: {baseline_lgb_mae:.4f} (피처 {len(feature_cols_all)}개)", flush=True)

# ============================================================
# 실험 1: 피처 선택 — 하위 30% 제거
# ============================================================
print("\n=== 실험 1: 피처 중요도 기반 선택 (하위 30% 제거) ===", flush=True)
importance = pd.DataFrame({
    'feature': feature_cols_all,
    'importance': np.mean([m.feature_importances_ for m in baseline_lgb_models], axis=0)
}).sort_values('importance', ascending=False)

cutoff = int(len(importance) * 0.30)
low_importance_features = importance.tail(cutoff)['feature'].tolist()
feature_cols_selected = [c for c in feature_cols_all if c not in low_importance_features]
print(f"  제거 피처 수: {cutoff}개, 남은 피처 수: {len(feature_cols_selected)}개", flush=True)
print(f"  제거된 피처 (하위 {cutoff}개): {low_importance_features[:10]}...", flush=True)

_, _, exp1_mae, _ = train_lgb_cv(
    X_all, y_log, y, folds_scenario, feature_cols_selected
)
print(f"  실험 1 LGB CV MAE: {exp1_mae:.4f} (피처 {len(feature_cols_selected)}개)", flush=True)
print(f"  기준선 대비 변화: {exp1_mae - baseline_lgb_mae:+.4f}", flush=True)

# ============================================================
# 실험 2: 정규화 강화
# ============================================================
print("\n=== 실험 2: 정규화 강화 (LightGBM) ===", flush=True)
reg_params = dict(
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=50,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5
)

_, _, exp2_mae, _ = train_lgb_cv(
    X_all, y_log, y, folds_scenario, feature_cols_selected, params_override=reg_params
)
print(f"  실험 2 LGB CV MAE: {exp2_mae:.4f}", flush=True)
print(f"  기준선 대비 변화: {exp2_mae - baseline_lgb_mae:+.4f}", flush=True)

# ============================================================
# 실험 3: layout 피처 추가 효과 (선택된 피처 + layout 파생 피처)
# ============================================================
print("\n=== 실험 3: layout 파생 피처 추가 효과 ===", flush=True)
new_layout_features = ['robot_per_packstation', 'charger_density', 'intersection_density',
                       'robot_compactness', 'dispersion_robot']
# 이미 feature_cols_all에 포함되어 있으므로, 선택된 피처에서 새 피처가 제거되었을 수 있음
# 새 layout 피처를 반드시 포함
feature_cols_with_layout = list(set(feature_cols_selected) | set(new_layout_features))
# 컬럼 순서 유지
feature_cols_with_layout = [c for c in feature_cols_all if c in feature_cols_with_layout]

_, _, exp3_mae, _ = train_lgb_cv(
    X_all, y_log, y, folds_scenario, feature_cols_with_layout, params_override=reg_params
)
print(f"  실험 3 LGB CV MAE: {exp3_mae:.4f} (피처 {len(feature_cols_with_layout)}개)", flush=True)
print(f"  기준선 대비 변화: {exp3_mae - baseline_lgb_mae:+.4f}", flush=True)

# ============================================================
# 실험 4: layout_id GroupKFold vs scenario_id GroupKFold
# ============================================================
print("\n=== 실험 4: layout_id GroupKFold 검증 ===", flush=True)
gkf_layout = GroupKFold(n_splits=5)
folds_layout = list(gkf_layout.split(X_all, y_log, groups_layout))

_, _, exp4_mae, _ = train_lgb_cv(
    X_all, y_log, y, folds_layout, feature_cols_with_layout, params_override=reg_params
)
print(f"  layout GroupKFold LGB CV MAE: {exp4_mae:.4f}", flush=True)
print(f"  scenario GroupKFold LGB CV MAE: {exp3_mae:.4f}", flush=True)

# 더 나은 검증 전략 선택
if exp4_mae > exp3_mae:
    print(f"  → layout GroupKFold MAE가 더 높음 ({exp4_mae:.4f} > {exp3_mae:.4f})", flush=True)
    print(f"  → layout GroupKFold가 unseen layout 상황을 더 보수적으로 평가", flush=True)
    print(f"  → layout GroupKFold를 최종 검증 전략으로 채택 (Public MAE와 갭 줄이기 목적)", flush=True)
    final_folds = folds_layout
    final_fold_name = "layout GroupKFold"
else:
    print(f"  → scenario GroupKFold MAE가 더 높거나 동일", flush=True)
    print(f"  → scenario GroupKFold 유지", flush=True)
    final_folds = folds_scenario
    final_fold_name = "scenario GroupKFold"

# ============================================================
# 5. 최종 앙상블 — 피처 선택 + 정규화 + layout 피처
# ============================================================
print(f"\n=== 최종 앙상블 (검증: {final_fold_name}) ===", flush=True)
final_features = feature_cols_with_layout
X_final = X_all[final_features]
X_test_final = X_test_all[final_features]

# --- LightGBM ---
print("  LightGBM 학습...", flush=True)
lgb_oof = np.zeros(len(X_final))
lgb_test = np.zeros(len(X_test_final))
lgb_maes = []

for fold, (tr_idx, va_idx) in enumerate(final_folds):
    Xtr, Xva = X_final.iloc[tr_idx], X_final.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=2000, learning_rate=0.03,
        num_leaves=63, random_state=42, n_jobs=-1, verbose=-1,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=50,
        feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    lgb_oof[va_idx] = pred
    lgb_test += np.clip(np.expm1(m.predict(X_test_final)), 0, None) / len(final_folds)
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    lgb_maes.append(mae)
    print(f"    LGB Fold {fold+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

lgb_cv = mean_absolute_error(y, lgb_oof)
print(f"  LightGBM CV MAE: {lgb_cv:.4f}", flush=True)

# --- CatBoost ---
print("  CatBoost 학습...", flush=True)
cat_oof = np.zeros(len(X_final))
cat_test = np.zeros(len(X_test_final))
cat_maes = []

for fold, (tr_idx, va_idx) in enumerate(final_folds):
    Xtr, Xva = X_final.iloc[tr_idx], X_final.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=6,
        loss_function='MAE', random_seed=42, verbose=0,
        l2_leaf_reg=5.0, min_data_in_leaf=50,
        subsample=0.7, colsample_bylevel=0.7
    )
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100)
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    cat_oof[va_idx] = pred
    cat_test += np.clip(np.expm1(m.predict(X_test_final)), 0, None) / len(final_folds)
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    cat_maes.append(mae)
    print(f"    Cat Fold {fold+1} MAE: {mae:.4f} (iter: {m.best_iteration_})", flush=True)

cat_cv = mean_absolute_error(y, cat_oof)
print(f"  CatBoost CV MAE: {cat_cv:.4f}", flush=True)

# --- XGBoost ---
print("  XGBoost 학습...", flush=True)
xgb_oof = np.zeros(len(X_final))
xgb_test = np.zeros(len(X_test_final))
xgb_maes = []

for fold, (tr_idx, va_idx) in enumerate(final_folds):
    Xtr, Xva = X_final.iloc[tr_idx], X_final.iloc[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    m = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        eval_metric='mae', tree_method='hist', random_state=42,
        verbosity=0, early_stopping_rounds=100,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=50,
        subsample=0.7, colsample_bytree=0.7
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred_log = m.predict(Xva)
    pred = np.clip(np.expm1(pred_log), 0, None)
    xgb_oof[va_idx] = pred
    xgb_test += np.clip(np.expm1(m.predict(X_test_final)), 0, None) / len(final_folds)
    mae = mean_absolute_error(y.iloc[va_idx], pred)
    xgb_maes.append(mae)
    print(f"    XGB Fold {fold+1} MAE: {mae:.4f} (iter: {m.best_iteration})", flush=True)

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
print(f"Ensemble CV MAE: {ens_cv:.4f}", flush=True)

# ============================================================
# 6. 결과 비교 테이블
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 3A 결과 ===", flush=True)
print("=" * 60, flush=True)
print(f"실험 0 (기준선):       LGB CV MAE {baseline_lgb_mae:.4f} (피처 {len(feature_cols_all)}개)", flush=True)
print(f"실험 1 (피처 선택):    LGB CV MAE {exp1_mae:.4f} ({len(feature_cols_all)}→{len(feature_cols_selected)}개 피처)", flush=True)
print(f"실험 2 (정규화 강화):  LGB CV MAE {exp2_mae:.4f}", flush=True)
print(f"실험 3 (layout 피처):  LGB CV MAE {exp3_mae:.4f} (피처 {len(feature_cols_with_layout)}개)", flush=True)
print(f"실험 4 (layout GKF):   LGB CV MAE {exp4_mae:.4f}", flush=True)
print(f"검증 전략: {final_fold_name}", flush=True)
print(f"최종 LightGBM CV MAE:  {lgb_cv:.4f}", flush=True)
print(f"최종 CatBoost CV MAE:  {cat_cv:.4f}", flush=True)
print(f"최종 XGBoost CV MAE:   {xgb_cv:.4f}", flush=True)
print(f"최종 앙상블 CV MAE:    {ens_cv:.4f}", flush=True)
print(f"Phase 2 앙상블:        8.8253", flush=True)
print(f"개선폭:                {ens_cv - 8.8253:+.4f}", flush=True)

# ============================================================
# 7. 제출 파일 생성
# ============================================================
print("\n=== 제출 파일 생성 ===", flush=True)
test_blended = best_w[0] * lgb_test + best_w[1] * cat_test + best_w[2] * xgb_test
test_blended = np.clip(test_blended, 0, None)

submission = sample_sub.copy()
submission['avg_delay_minutes_next_30m'] = test_blended
submission.to_csv('output/submission_phase3a.csv', index=False)

assert list(submission.columns) == list(sample_sub.columns), "컬럼 불일치!"
assert len(submission) == len(sample_sub), "행 수 불일치!"
assert (submission['ID'] == sample_sub['ID']).all(), "ID 순서 불일치!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "음수 예측!"

print("submission_phase3a.csv 생성 완료", flush=True)
print(submission.describe(), flush=True)

# ============================================================
# 8. 피처 중요도 시각화
# ============================================================
print("\n=== 피처 중요도 시각화 ===", flush=True)
# 최종 앙상블에서 사용한 LGB 모델 기준은 아니지만, baseline 모델의 중요도로 선택 근거 시각화
top30 = importance.head(30).sort_values('importance')
ts_suffixes = ['_lag', '_roll', '_diff1', '_cumsum']
layout_new = ['robot_per_packstation', 'charger_density', 'intersection_density',
              'robot_compactness', 'dispersion_robot']

def get_color(f):
    if f in layout_new:
        return 'forestgreen'
    elif any(s in f for s in ts_suffixes):
        return 'coral'
    else:
        return 'steelblue'

colors = [get_color(f) for f in top30['feature']]

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top30['feature'], top30['importance'], color=colors)
ax.set_title('피처 중요도 Top 30 (Phase 3A)')
ax.set_xlabel('중요도')
ax.legend(handles=[
    Patch(color='coral', label='시계열 피처'),
    Patch(color='steelblue', label='기존 피처'),
    Patch(color='forestgreen', label='신규 layout 피처')
])
plt.tight_layout()
plt.savefig('output/feature_importance_phase3a.png', dpi=150, bbox_inches='tight')
print("feature_importance_phase3a.png 저장 완료", flush=True)

print("\n=== Phase 3A 완료 ===", flush=True)
