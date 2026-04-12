import pandas as pd
import numpy as np
import pickle
import os
import gc
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)

print("=" * 70, flush=True)
print("Phase 23 Track A v2: Layout-aware, Scenario-agnostic", flush=True)
print("=" * 70, flush=True)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')
train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

TARGET = 'avg_delay_minutes_next_30m'
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

# ##############################################################
# Step 1: Row-level Saturation Features
# ##############################################################
print("\nStep 1: Row-level Saturation Features", flush=True)


def saturation_features(df):
    # Pack saturation
    df['pack_util_margin'] = 1.0 - df['pack_utilization']
    df['pack_util_sat_85'] = (df['pack_utilization'] > 0.85).astype(int)
    df['pack_util_sat_95'] = (df['pack_utilization'] > 0.95).astype(int)

    # Robot saturation
    df['robot_util_margin'] = 1.0 - df['robot_utilization']
    df['robot_active_ratio'] = df['robot_active'] / (df['robot_total'] + 1)
    df['robot_active_margin'] = 1.0 - df['robot_active_ratio']
    df['robot_util_sat_80'] = (df['robot_utilization'] > 0.8).astype(int)

    # Multi-resource pressure (cascading trigger)
    df['pack_AND_robot_sat'] = (
        (df['pack_utilization'] > 0.85).astype(int) *
        (df['robot_utilization'] > 0.7).astype(int)
    )
    df['pack_robot_combined'] = df['pack_utilization'] * df['robot_utilization']

    # Inflow pressure
    df['inflow_vs_pack'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['inflow_vs_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['inflow_vs_robot_total'] = df['order_inflow_15m'] / (df['robot_total'] + 1)

    # Dock
    df['dock_util_sat'] = (df['loading_dock_util'] > 0.8).astype(int)

    return df


train = saturation_features(train)
test = saturation_features(test)
print(f"  Features added: ~13", flush=True)

# ##############################################################
# Step 2: Queueing W Features
# ##############################################################
print("\nStep 2: Queueing W Features", flush=True)


def queueing_features(df):
    eps = 1e-6

    df['W_pack'] = df['pack_utilization'] / (1 - df['pack_utilization'] + eps)
    df['W_robot'] = df['robot_utilization'] / (1 - df['robot_utilization'] + eps)

    rho_r = df['robot_active'] / (df['robot_total'] + 1)
    df['W_robot_active'] = rho_r / (1 - rho_r + eps)
    df['W_dock'] = df['loading_dock_util'] / (1 - df['loading_dock_util'] + eps)

    # Little's Law: L = lambda * W
    df['L_pack'] = df['order_inflow_15m'] * df['W_pack']
    df['L_robot'] = df['order_inflow_15m'] * df['W_robot']

    # Bottleneck
    df['W_max'] = df[['W_pack', 'W_robot', 'W_dock']].max(axis=1)

    return df


train = queueing_features(train)
test = queueing_features(test)
print(f"  Features added: ~7", flush=True)

# ##############################################################
# Step 3: Layout Capacity Features
# ##############################################################
print("\nStep 3: Layout Capacity Features", flush=True)


def layout_capacity(df):
    # Density
    df['robot_per_area'] = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['charger_per_robot'] = df['charger_count'] / (df['robot_total'] + 1)
    df['pack_per_area'] = df['pack_station_count'] / (df['floor_area_sqm'] + 1)
    df['aisle_robot_density'] = df['robot_total'] / (df['aisle_width_avg'] + 0.1)

    # Congestion potential
    df['congestion_potential'] = (
        df['aisle_robot_density'] * df['robot_active'] / (df['robot_total'] + 1)
    )

    # Layout efficiency
    df['oneway_penalty'] = df['one_way_ratio'] * df['layout_compactness']
    df['intersection_per_robot'] = df['intersection_count'] / (df['robot_total'] + 1)

    return df


train = layout_capacity(train)
test = layout_capacity(test)
print(f"  Features added: ~7", flush=True)

# ##############################################################
# Step 4: Layout x Operation Interactions
# ##############################################################
print("\nStep 4: Layout x Operation Interactions", flush=True)


def layout_operation_interaction(df):
    # Aisle width x robot traffic
    df['aisle_x_robot_active'] = df['aisle_width_avg'] * df['robot_active']
    df['aisle_x_inflow'] = df['aisle_width_avg'] * df['order_inflow_15m']
    df['narrow_aisle_pressure'] = df['robot_active'] / (df['aisle_width_avg'] + 0.1)

    # Intersection x congestion
    df['intersection_x_congestion'] = df['intersection_count'] * df['congestion_score']
    df['intersection_x_collision'] = df['intersection_count'] * df['near_collision_15m']

    # Charger saturation
    df['charger_demand'] = df['robot_charging'] / (df['charger_count'] + 1)
    df['charger_queue_per_charger'] = df['charge_queue_length'] / (df['charger_count'] + 1)

    # Pack station pressure
    df['pack_station_demand'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)

    # Zone dispersion x traffic
    df['zone_x_traffic'] = df['zone_dispersion'] * df['robot_active']

    # Layout compactness x robot
    df['compactness_x_robot'] = df['layout_compactness'] * df['robot_active']

    # Ceiling height x inflow
    df['vertical_x_inflow'] = df['ceiling_height_m'] * df['order_inflow_15m']

    # Building age x fault
    df['age_x_fault'] = df['building_age_years'] * df['fault_count_15m']

    return df


train = layout_operation_interaction(train)
test = layout_operation_interaction(test)
print(f"  Features added: ~12", flush=True)

# ##############################################################
# Step 5: Within-Layout Percentile
# ##############################################################
print("\nStep 5: Within-Layout Percentile", flush=True)

key_operation_cols = [
    'order_inflow_15m', 'robot_active', 'robot_utilization',
    'charge_queue_length', 'congestion_score', 'max_zone_density',
    'pack_utilization', 'fault_count_15m', 'near_collision_15m',
    'blocked_path_15m', 'loading_dock_util'
]

# Train + Test combined (unsupervised, no target leakage)
combined = pd.concat([
    train.assign(source='train'),
    test.assign(source='test')
], axis=0, ignore_index=True)

for col in key_operation_cols:
    combined[f'{col}_layout_pct'] = combined.groupby('layout_id')[col].rank(pct=True)

# Split back
train_new = combined[combined['source'] == 'train'].drop(columns=['source']).reset_index(drop=True)
test_new = combined[combined['source'] == 'test'].drop(columns=['source']).reset_index(drop=True)

assert len(train_new) == len(train), f"Train size mismatch: {len(train_new)} vs {len(train)}"
assert len(test_new) == len(test), f"Test size mismatch: {len(test_new)} vs {len(test)}"

train = train_new
test = test_new
del combined, train_new, test_new
gc.collect()

print(f"  Percentile features added: {len(key_operation_cols)}", flush=True)

# ##############################################################
# Step 6: Layout Type Encoding
# ##############################################################
print("\nStep 6: Layout Type Encoding", flush=True)

# One-hot
for lt in ['grid', 'hybrid', 'hub_spoke', 'narrow']:
    train[f'is_{lt}'] = (train['layout_type'] == lt).astype(int)
    test[f'is_{lt}'] = (test['layout_type'] == lt).astype(int)


# Target encoding (KFold)
def oof_target_encode_simple(train_df, test_df, target_col, encode_col, n_splits=5, smoothing=10):
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


train['layout_type_te'], test['layout_type_te'] = oof_target_encode_simple(
    train, test, TARGET, 'layout_type', n_splits=5, smoothing=5
)

print(f"  Encoded layout_type", flush=True)

# ##############################################################
# Step 7: Feature Selection (3-fold importance)
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 7: Feature Selection", flush=True)
print("=" * 70, flush=True)

drop_cols = ['ID', 'scenario_id', 'layout_id', 'layout_type', TARGET]
feature_cols = [c for c in train.columns if c not in drop_cols]
numeric_cols = [c for c in feature_cols if train[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
feature_cols = numeric_cols
print(f"  Total numeric features: {len(feature_cols)}", flush=True)

X = train[feature_cols]
y = train[TARGET]
groups = train['layout_id']

importance_folds = []
gkf_imp = GroupKFold(n_splits=3)

for fold_idx, (tr_idx, val_idx) in enumerate(gkf_imp.split(X, y, groups)):
    quick = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, feature_fraction=0.7,
        objective='huber', random_state=42 + fold_idx,
        verbose=-1, n_jobs=-1
    )
    quick.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

    importance_folds.append(pd.DataFrame({
        'feature': feature_cols,
        'importance': quick.feature_importances_
    }))
    print(f"  Fold {fold_idx + 1} importance computed", flush=True)

imp = pd.concat(importance_folds).groupby('feature')['importance'].mean().reset_index()
imp = imp.sort_values('importance', ascending=False)
imp.to_csv('output/phase23_track_a_v2_importance.csv', index=False)

# Removal: ALL folds zero OR bottom 5%
zero_imp_all = set(feature_cols)
for fold_imp in importance_folds:
    non_zero = set(fold_imp[fold_imp['importance'] > 0]['feature'])
    zero_imp_all -= non_zero

low_threshold = imp['importance'].quantile(0.05)
low_imp = set(imp[imp['importance'] <= low_threshold]['feature'].tolist())
remove_features = zero_imp_all | low_imp
final_features = [f for f in feature_cols if f not in remove_features]

print(f"\n  Zero in ALL folds: {len(zero_imp_all)}", flush=True)
print(f"  Bottom 5%: {len(low_imp)}", flush=True)
print(f"  Removing: {len(remove_features)}", flush=True)
print(f"  Final: {len(final_features)}", flush=True)
print(f"  Top 15: {imp.head(15)['feature'].tolist()}", flush=True)

with open('output/phase23_track_a_v2_features.pkl', 'wb') as f:
    pickle.dump({
        'final_features': final_features,
        'all_features': feature_cols,
        'importance': imp.to_dict('records'),
        'zero_imp_all_folds': list(zero_imp_all),
        'low_imp': list(low_imp)
    }, f)

# ##############################################################
# Step 8: 5-Fold CV (GroupKFold by layout_id)
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("Step 8: 5-Fold CV (GroupKFold by layout_id)", flush=True)
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

    try:
        model = lgb.LGBMRegressor(
            n_estimators=5000, learning_rate=0.03, num_leaves=127,
            min_child_samples=30, feature_fraction=0.6, bagging_fraction=0.8,
            bagging_freq=5, objective='huber', alpha=0.9,
            random_state=42, verbose=-1, n_jobs=-1, device='gpu'
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    except Exception as e:
        print(f"    GPU failed: {e}, CPU fallback", flush=True)
        model = lgb.LGBMRegressor(
            n_estimators=5000, learning_rate=0.03, num_leaves=127,
            min_child_samples=30, feature_fraction=0.6, bagging_fraction=0.8,
            bagging_freq=5, objective='huber', alpha=0.9,
            random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test_final) / 5

    fold_mae = mean_absolute_error(y_val, oof_preds[val_idx])
    fold_maes.append(fold_mae)
    print(f"    Fold {fold + 1} MAE: {fold_mae:.4f}", flush=True)

cv_mae = mean_absolute_error(y, oof_preds)
print(f"\n  CV MAE: {cv_mae:.4f}", flush=True)
print(f"  Fold MAEs: {[f'{m:.4f}' for m in fold_maes]}", flush=True)
print(f"  Fold std: {np.std(fold_maes):.4f}", flush=True)
print(f"\n  v1 baseline (GroupKFold): 8.7910", flush=True)
print(f"  v2 change: {cv_mae - 8.7910:+.4f}", flush=True)
print(f"  (Note: v1 had scenario aggregate leakage issue)", flush=True)

# ##############################################################
# Step 9: Save Results
# ##############################################################
print("\nStep 9: Save Results", flush=True)

test_preds = np.clip(test_preds, 0, train[TARGET].max() * 1.1)
submission = pd.read_csv('data/sample_submission.csv')
submission[TARGET] = test_preds
submission.to_csv('output/phase23_track_a_v2_submission.csv', index=False)

np.save('output/phase23_track_a_v2_oof.npy', oof_preds)
np.save('output/phase23_track_a_v2_test.npy', test_preds)

with open('output/phase23_track_a_v2_ckpt.pkl', 'wb') as f:
    pickle.dump({
        'cv_mae': cv_mae,
        'fold_maes': fold_maes,
        'final_features': final_features,
        'feature_count': len(final_features)
    }, f)

print(f"\nTrack A v2 complete", flush=True)
print(f"   CV MAE: {cv_mae:.4f}", flush=True)
print(f"   Features: {len(final_features)}", flush=True)
print(f"\n   -> Submit and compare with v1 Public 10.28, baseline 9.86", flush=True)
