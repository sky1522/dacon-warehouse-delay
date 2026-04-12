import pandas as pd
import numpy as np
import pickle
import os
import gc
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)

print("=" * 70, flush=True)
print("Phase 23 Track A: AMEX-style Aggregate Features + Feature Selection", flush=True)
print("=" * 70, flush=True)

# Data loading
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')

# Merge layout
train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

TARGET = 'avg_delay_minutes_next_30m'
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

# ##############################################################
# Step 1: 핵심 변수 정의
# ##############################################################

# Numerical operation columns (scenario 내 변동하는 것)
OPERATION_COLS = [
    'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order',
    'urgent_order_ratio', 'heavy_item_ratio', 'cold_chain_ratio',
    'sku_concentration', 'robot_active', 'robot_idle', 'robot_charging',
    'robot_utilization', 'avg_trip_distance', 'task_reassign_15m',
    'battery_mean', 'battery_std', 'low_battery_ratio',
    'charge_queue_length', 'avg_charge_wait', 'congestion_score',
    'max_zone_density', 'blocked_path_15m', 'near_collision_15m',
    'fault_count_15m', 'avg_recovery_time', 'replenishment_overlap',
    'pack_utilization', 'manual_override_ratio', 'loading_dock_util',
    'staff_on_floor', 'forklift_active_count', 'conveyor_speed_mps',
    'prev_shift_volume', 'order_wave_count', 'pick_list_length_avg',
    'express_lane_util', 'bulk_order_ratio', 'staging_area_util',
    'warehouse_temp_avg', 'humidity_pct', 'external_temp_c',
    'wind_speed_kmh', 'precipitation_mm', 'lighting_level_lux',
    'ambient_noise_db', 'co2_level_ppm', 'hvac_power_kw',
    'wms_response_time_ms', 'scanner_error_rate', 'network_latency_ms',
    'label_print_queue', 'outbound_truck_wait_min', 'backorder_ratio',
    'shift_handover_delay_min'
]

# Layout constant columns (scenario 내 불변)
LAYOUT_COLS = [
    'aisle_width_avg', 'intersection_count', 'one_way_ratio',
    'pack_station_count', 'charger_count', 'layout_compactness',
    'zone_dispersion', 'robot_total', 'building_age_years',
    'floor_area_sqm', 'ceiling_height_m', 'fire_sprinkler_count',
    'emergency_exit_count'
]

# Filter to existing columns
OPERATION_COLS = [c for c in OPERATION_COLS if c in train.columns]
LAYOUT_COLS = [c for c in LAYOUT_COLS if c in train.columns]

print(f"Operation cols: {len(OPERATION_COLS)}", flush=True)
print(f"Layout cols: {len(LAYOUT_COLS)}", flush=True)

# ##############################################################
# Step 2: Scenario-level Aggregate Features (AMEX 2위 스타일)
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 2: Scenario Aggregate Features", flush=True)
print("=" * 70, flush=True)


def create_scenario_aggregates(df, cols):
    """
    AMEX 2위 스타일 aggregate.
    각 scenario의 통계를 모든 row에 broadcast.
    """
    agg_funcs = ['mean', 'std', 'min', 'max', 'median']
    aggs = df.groupby('scenario_id')[cols].agg(agg_funcs)

    # Flatten multi-index columns
    aggs.columns = [f'{col}_scn_{func}' for col, func in aggs.columns]

    # P90, P10 (별도 계산)
    p90 = df.groupby('scenario_id')[cols].quantile(0.9)
    p90.columns = [f'{col}_scn_p90' for col in p90.columns]
    p10 = df.groupby('scenario_id')[cols].quantile(0.1)
    p10.columns = [f'{col}_scn_p10' for col in p10.columns]

    # Range (max - min)
    for col in cols:
        aggs[f'{col}_scn_range'] = aggs[f'{col}_scn_max'] - aggs[f'{col}_scn_min']

    # Merge
    result = aggs.join(p90).join(p10).reset_index()
    return result


print("  Creating train aggregates...", flush=True)
train_scn_agg = create_scenario_aggregates(train, OPERATION_COLS)
print(f"  Train aggregate shape: {train_scn_agg.shape}", flush=True)

print("  Creating test aggregates...", flush=True)
test_scn_agg = create_scenario_aggregates(test, OPERATION_COLS)
print(f"  Test aggregate shape: {test_scn_agg.shape}", flush=True)

# Merge back to row-level
train = train.merge(train_scn_agg, on='scenario_id', how='left')
test = test.merge(test_scn_agg, on='scenario_id', how='left')

print(f"\n  After scenario agg - Train: {train.shape}", flush=True)
print(f"  Added {len(train_scn_agg.columns) - 1} scenario aggregate features", flush=True)

del train_scn_agg, test_scn_agg
gc.collect()

# ##############################################################
# Step 3: Saturation Features (Bin 9 대응)
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 3: Saturation Features (Bin 9 대응)", flush=True)
print("=" * 70, flush=True)


def create_saturation_features(df):
    """
    EDA 4 결과: pack_util, robot_util이 1.0에 포화될 때 cascading
    """
    # Pack saturation
    df['pack_util_sat_90'] = (df['pack_utilization'] > 0.9).astype(int)
    df['pack_util_sat_95'] = (df['pack_utilization'] > 0.95).astype(int)
    df['pack_util_margin'] = 1.0 - df['pack_utilization']

    # Robot saturation
    df['robot_util_sat_80'] = (df['robot_utilization'] > 0.8).astype(int)
    df['robot_util_margin'] = 1.0 - df['robot_utilization']
    df['robot_active_ratio'] = df['robot_active'] / (df['robot_total'] + 1)
    df['robot_active_margin'] = 1.0 - df['robot_active_ratio']

    # Multi-resource saturation (cascading trigger)
    df['pack_AND_robot_sat'] = (
        (df['pack_utilization'] > 0.85).astype(int) *
        (df['robot_utilization'] > 0.7).astype(int)
    )
    df['pack_robot_combined_pressure'] = (
        df['pack_utilization'] * df['robot_utilization']
    )

    # Order inflow pressure
    df['inflow_vs_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['inflow_vs_pack'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)

    # Loading dock saturation
    df['dock_util_sat'] = (df['loading_dock_util'] > 0.8).astype(int)

    return df


train = create_saturation_features(train)
test = create_saturation_features(test)

print(f"  Saturation features added: ~15", flush=True)
print(f"  Train shape: {train.shape}", flush=True)

# ##############################################################
# Step 4: Queueing Theory Features
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 4: Queueing Theory Features", flush=True)
print("=" * 70, flush=True)


def create_queueing_features(df):
    """
    Pollaczek-Khinchine: Wait time = rho / (1 - rho) in M/M/1 queue
    """
    eps = 1e-6

    # Pack station queue (W_pack)
    df['W_pack'] = df['pack_utilization'] / (1 - df['pack_utilization'] + eps)
    df['W_pack_squared'] = df['W_pack'] ** 2

    # Robot queue (W_robot)
    df['W_robot'] = df['robot_utilization'] / (1 - df['robot_utilization'] + eps)

    # Robot active queue (from active/total ratio)
    rho_r = df['robot_active'] / (df['robot_total'] + 1)
    df['W_robot_active'] = rho_r / (1 - rho_r + eps)

    # Dock queue
    df['W_dock'] = df['loading_dock_util'] / (1 - df['loading_dock_util'] + eps)

    # Total queueing time (sum of bottlenecks)
    df['W_total'] = df[['W_pack', 'W_robot', 'W_dock']].sum(axis=1)
    df['W_max'] = df[['W_pack', 'W_robot', 'W_dock']].max(axis=1)

    # Little's Law approximation: L = lambda * W
    df['L_pack'] = df['order_inflow_15m'] * df['W_pack']
    df['L_robot'] = df['order_inflow_15m'] * df['W_robot']

    return df


train = create_queueing_features(train)
test = create_queueing_features(test)

print(f"  Queueing features added: ~10", flush=True)

# ##############################################################
# Step 5: Layout-aware Features
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 5: Layout-aware Features", flush=True)
print("=" * 70, flush=True)

# Layout capacity ratios
for df in [train, test]:
    df['robot_per_area'] = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['charger_per_robot'] = df['charger_count'] / (df['robot_total'] + 1)
    df['pack_per_area'] = df['pack_station_count'] / (df['floor_area_sqm'] + 1)
    df['aisle_robot_density'] = df['robot_total'] / (df['aisle_width_avg'] + 0.1)

    # Layout type one-hot
    for lt in ['grid', 'hybrid', 'hub_spoke', 'narrow']:
        df[f'is_{lt}'] = (df['layout_type'] == lt).astype(int)

print(f"  Layout features added: ~8", flush=True)

# ##############################################################
# Step 6: OOF Target Encoding (layout_type only)
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 6: OOF Target Encoding", flush=True)
print("=" * 70, flush=True)


def oof_target_encode_simple(train_df, test_df, target_col, encode_col, n_splits=5, smoothing=10):
    """
    Simple KFold target encoding with Bayesian smoothing.
    (GroupKFold by encode_col causes leakage issues for layout_id,
     and ValueError for layout_type with only 4 unique values.)
    """
    from sklearn.model_selection import KFold
    global_mean = train_df[target_col].mean()
    train_encoded = np.zeros(len(train_df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        stats = tr_fold.groupby(encode_col)[target_col].agg(['mean', 'count'])
        smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        train_encoded[val_idx] = train_df.iloc[val_idx][encode_col].map(smooth).fillna(global_mean).values

    stats_full = train_df.groupby(encode_col)[target_col].agg(['mean', 'count'])
    smooth_full = (stats_full['mean'] * stats_full['count'] + global_mean * smoothing) / (stats_full['count'] + smoothing)
    test_encoded = test_df[encode_col].map(smooth_full).fillna(global_mean).values

    return train_encoded, test_encoded


# layout_id target encoding 제거 (GroupKFold 설계 결함: val layout이 train에 없어 전부 global_mean)
print("  Skipping layout_id target encoding (GroupKFold design issue)", flush=True)
print("  Layout info already captured via layout_info.csv features", flush=True)

# layout_type만 encoding (simple KFold)
print("  Encoding layout_type...", flush=True)
train['layout_type_te'], test['layout_type_te'] = oof_target_encode_simple(
    train, test, TARGET, 'layout_type', n_splits=5, smoothing=5
)

print("  Target encoding done", flush=True)

# ##############################################################
# Step 7: NaN Pattern Features
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 7: NaN Pattern Features", flush=True)
print("=" * 70, flush=True)


def create_nan_features(df, cols):
    """NaN은 정보. 패턴을 feature로."""
    # Row-level NaN count
    df['nan_count_operation'] = df[cols].isna().sum(axis=1)
    df['nan_ratio_operation'] = df['nan_count_operation'] / len(cols)

    # Scenario-level NaN pattern
    df['nan_count_scn_mean'] = df.groupby('scenario_id')['nan_count_operation'].transform('mean')
    df['nan_count_scn_max'] = df.groupby('scenario_id')['nan_count_operation'].transform('max')

    # 주요 컬럼 개별 flag
    critical_cols = ['order_inflow_15m', 'robot_active', 'pack_utilization',
                     'charge_queue_length', 'congestion_score']
    for col in critical_cols:
        if col in df.columns:
            df[f'{col}_isnan'] = df[col].isna().astype(int)

    return df


train = create_nan_features(train, OPERATION_COLS)
test = create_nan_features(test, OPERATION_COLS)

print(f"  NaN features added: ~10", flush=True)

# ##############################################################
# Step 8: Feature Selection — Zero Importance Removal
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 8: Multi-fold Feature Selection", flush=True)
print("=" * 70, flush=True)

# Prepare features
drop_cols = ['ID', 'scenario_id', 'layout_id', 'layout_type', TARGET]
feature_cols = [c for c in train.columns if c not in drop_cols]

# Keep only numeric
numeric_cols = []
for c in feature_cols:
    if train[c].dtype in [np.float64, np.int64, np.float32, np.int32]:
        numeric_cols.append(c)
feature_cols = numeric_cols
print(f"  Total numeric features: {len(feature_cols)}", flush=True)

X = train[feature_cols]
y = train[TARGET]
groups = train['layout_id']

# 3-fold importance (different seeds for robustness)
importance_folds = []
gkf_imp = GroupKFold(n_splits=3)

for fold_idx, (tr_idx, val_idx) in enumerate(gkf_imp.split(X, y, groups)):
    quick_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, feature_fraction=0.7,
        objective='huber', random_state=42 + fold_idx,
        verbose=-1, n_jobs=-1
    )
    quick_model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
    )
    fold_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': quick_model.feature_importances_
    })
    importance_folds.append(fold_imp)
    print(f"  Fold {fold_idx + 1} importance computed", flush=True)

imp = pd.concat(importance_folds).groupby('feature')['importance'].mean().reset_index()
imp = imp.sort_values('importance', ascending=False)
imp.to_csv('output/phase23_track_a_importance.csv', index=False)

# Removal criterion 1: zero in ALL folds
zero_imp_all = set(feature_cols)
for fold_imp in importance_folds:
    non_zero = set(fold_imp[fold_imp['importance'] > 0]['feature'])
    zero_imp_all -= non_zero
print(f"\n  Zero in ALL folds: {len(zero_imp_all)}", flush=True)

# Removal criterion 2: bottom 5% average importance
low_threshold = imp['importance'].quantile(0.05)
low_imp = set(imp[imp['importance'] <= low_threshold]['feature'].tolist())
print(f"  Bottom 5% avg importance: {len(low_imp)}", flush=True)

# Union
remove_features = zero_imp_all | low_imp
final_features = [f for f in feature_cols if f not in remove_features]

print(f"\n  Removing: {len(remove_features)} features", flush=True)
print(f"  Final: {len(final_features)}", flush=True)
print(f"  Top 10: {imp.head(10)['feature'].tolist()}", flush=True)

# Save
with open('output/phase23_track_a_features.pkl', 'wb') as f:
    pickle.dump({
        'all_features': feature_cols,
        'zero_imp_all_folds': list(zero_imp_all),
        'low_imp': list(low_imp),
        'remove_features': list(remove_features),
        'final_features': final_features,
        'importance': imp.to_dict('records')
    }, f)

# ##############################################################
# Step 9: 5-Fold CV with Final Features
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 9: 5-Fold CV with Track A Features", flush=True)
print("=" * 70, flush=True)

X = train[final_features]
y = train[TARGET]
X_test_final = test[final_features]
groups = train['layout_id']

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_maes = []

gkf = GroupKFold(n_splits=5)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n  Fold {fold + 1}/5", flush=True)
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=30,
        feature_fraction=0.6,
        bagging_fraction=0.8,
        bagging_freq=5,
        objective='huber',
        alpha=0.9,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
        device='gpu'  # Kaggle T4
    )

    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        )
    except Exception as e:
        # GPU 실패 시 CPU fallback
        print(f"    GPU failed, CPU fallback: {e}", flush=True)
        model = lgb.LGBMRegressor(
            n_estimators=5000, learning_rate=0.03, num_leaves=127,
            min_child_samples=30, feature_fraction=0.6, bagging_fraction=0.8,
            bagging_freq=5, objective='huber', alpha=0.9,
            random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test_final) / 5

    fold_mae = mean_absolute_error(y_val, oof_preds[val_idx])
    fold_maes.append(fold_mae)
    print(f"    Fold {fold + 1} MAE: {fold_mae:.4f}", flush=True)

cv_mae = mean_absolute_error(y, oof_preds)
print(f"\n  CV MAE: {cv_mae:.4f}", flush=True)
print(f"  Fold MAEs: {[f'{m:.4f}' for m in fold_maes]}", flush=True)
print(f"  Fold std: {np.std(fold_maes):.4f}", flush=True)

# ##############################################################
# Step 10: Save Results
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 10: Save Results", flush=True)
print("=" * 70, flush=True)

# Clip predictions to valid range
test_preds = np.clip(test_preds, 0, train[TARGET].max() * 1.1)

# Submission
submission = pd.read_csv('data/sample_submission.csv')
submission[TARGET] = test_preds
submission.to_csv('output/phase23_track_a_submission.csv', index=False)

# OOF save (for later stacking)
np.save('output/phase23_track_a_oof.npy', oof_preds)
np.save('output/phase23_track_a_test.npy', test_preds)

# Save checkpoint
ckpt = {
    'cv_mae': cv_mae,
    'fold_maes': fold_maes,
    'final_features': final_features,
    'feature_count': len(final_features),
    'model_params': model.get_params()
}
with open('output/phase23_track_a_ckpt.pkl', 'wb') as f:
    pickle.dump(ckpt, f)

print(f"\nTrack A complete", flush=True)
print(f"   CV MAE: {cv_mae:.4f}", flush=True)
print(f"   Features: {len(final_features)}", flush=True)
print(f"   Files: output/phase23_track_a_*", flush=True)
