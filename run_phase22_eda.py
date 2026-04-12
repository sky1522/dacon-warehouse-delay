import pandas as pd
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output/phase22_eda', exist_ok=True)

# ##############################################################
# Data Loading
# ##############################################################
print("=" * 60, flush=True)
print("=== Phase 22 Pre-EDA: Cascading + Layout Cluster ===", flush=True)
print("=" * 60, flush=True)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout_info = pd.read_csv('data/layout_info.csv')
print(f"Train: {len(train)}, Test: {len(test)}", flush=True)
print(f"Train cols: {train.columns.tolist()[:10]}...", flush=True)
print(f"Layout cols: {layout_info.columns.tolist()}", flush=True)

# Merge layout features
train = train.merge(layout_info, on='layout_id', how='left')
test = test.merge(layout_info, on='layout_id', how='left')
print(f"After merge: train shape {train.shape}, test shape {test.shape}", flush=True)

# Position 생성 (scenario_id 내 순서)
train = train.sort_values(['scenario_id'])
train['position_in_scenario'] = train.groupby('scenario_id').cumcount()
test['position_in_scenario'] = test.groupby('scenario_id').cumcount()

# ##############################################################
# Q1: Cascading Detector Validation
# ##############################################################
print("\n=== Q1: Cascading Detector Correlation ===", flush=True)

# 1. rho utilization
train['rho_robot'] = train['robot_active'] / (train['robot_total'] + 1e-6)
train['rho_pack'] = train['pack_utilization']
train['rho_charger'] = train['charge_queue_length'] / (train['charger_count'] + 1e-6)
train['rho_max'] = train[['rho_robot', 'rho_pack', 'rho_charger']].max(axis=1)

# 2. Cascading binary indicators
train['rho_over_70'] = (train['rho_max'] > 0.70).astype(int)
train['rho_over_85'] = (train['rho_max'] > 0.85).astype(int)
train['rho_over_95'] = (train['rho_max'] > 0.95).astype(int)

# 3. Multi-resource pressure
train['robot_pressure'] = (train['rho_robot'] > 0.85).astype(int)
train['pack_pressure'] = (train['pack_utilization'] > 0.80).astype(int)
train['charger_pressure'] = (train['charge_queue_length'] > 5).astype(int)
train['multi_pressure'] = train['robot_pressure'] + train['pack_pressure'] + train['charger_pressure']

# 4. rho velocity (scenario 내 변화율)
train = train.sort_values(['scenario_id', 'position_in_scenario'])
train['rho_diff'] = train.groupby('scenario_id')['rho_max'].diff().fillna(0)
train['rho_velocity_3'] = train.groupby('scenario_id')['rho_max'].rolling(3).apply(
    lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 3 else 0
).reset_index(level=0, drop=True).fillna(0)

# 5. Target correlation
target = train['avg_delay_minutes_next_30m']
cascading_features = ['rho_max', 'rho_over_70', 'rho_over_85', 'rho_over_95',
                       'multi_pressure', 'rho_diff', 'rho_velocity_3']
corr_results = []
for f in cascading_features:
    corr = train[f].corr(target)
    print(f"  {f}: corr = {corr:.4f}", flush=True)
    corr_results.append({'feature': f, 'correlation': corr})

pd.DataFrame(corr_results).to_csv('output/phase22_eda/cascading_correlations.csv', index=False)

# 6. rho-band별 target 통계
print("\n=== Q1-2: Target by rho-band ===", flush=True)
bands = pd.cut(train['rho_max'], bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0])
band_stats = train.groupby(bands)['avg_delay_minutes_next_30m'].agg(['count', 'mean', 'median', 'max', 'std'])
print(band_stats, flush=True)

# ##############################################################
# Q2: Layout Cluster Validation
# ##############################################################
print("\n=== Q2: Layout Clustering ===", flush=True)

layout_features = ['aisle_width_avg', 'robot_total', 'intersection_count',
                    'layout_compactness', 'pack_station_count', 'charger_count',
                    'one_way_ratio', 'floor_area_sqm', 'zone_dispersion']

# layout_info에서 직접 KMeans
X_layouts = layout_info[layout_features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_layouts)
km = KMeans(n_clusters=10, random_state=42, n_init=10)
layout_info['cluster'] = km.fit_predict(X_scaled)
print(f"Total unique layouts: {len(layout_info)}", flush=True)

# Train/Test layout cluster 분포
train_layout_ids = train['layout_id'].unique()
test_layout_ids = test['layout_id'].unique()
train_clusters = layout_info[layout_info['layout_id'].isin(train_layout_ids)].set_index('layout_id')['cluster']
test_clusters = layout_info[layout_info['layout_id'].isin(test_layout_ids)].set_index('layout_id')['cluster']

print("\nCluster 분포:", flush=True)
print(f"  Train layouts ({len(train_layout_ids)}): {train_clusters.value_counts().sort_index().to_dict()}", flush=True)
print(f"  Test layouts ({len(test_layout_ids)}):  {test_clusters.value_counts().sort_index().to_dict()}", flush=True)

# Test/Train ratio per cluster
combined_cluster = pd.DataFrame({
    'train': train_clusters.value_counts().sort_index(),
    'test': test_clusters.value_counts().sort_index()
}).fillna(0)
combined_cluster['test_train_ratio'] = combined_cluster['test'] / (combined_cluster['train'] + 1)
print("\nTest/Train ratio per cluster:", flush=True)
print(combined_cluster.sort_values('test_train_ratio', ascending=False), flush=True)

# Cluster별 target 분포
train['layout_cluster'] = train['layout_id'].map(train_clusters)
print("\n=== Cluster별 Target 통계 ===", flush=True)
cluster_target = train.groupby('layout_cluster')['avg_delay_minutes_next_30m'].agg(['count', 'mean', 'median', 'std'])
print(cluster_target, flush=True)

# Save layout clusters
layout_info.to_csv('output/phase22_eda/layout_clusters.csv', index=False)

# Phase 16 OOF MAE per cluster
if os.path.exists('output/ckpt_phase16_lgb_huber.pkl'):
    with open('output/ckpt_phase16_lgb_huber.pkl', 'rb') as f:
        ckpt = pickle.load(f)
    train['phase16_oof'] = ckpt['oof']
    train['mae'] = np.abs(train['phase16_oof'] - train['avg_delay_minutes_next_30m'])
    cluster_mae = train.groupby('layout_cluster')['mae'].agg(['count', 'mean', 'median']).sort_values('mean', ascending=False)
    print("\n=== Cluster별 Phase 16 MAE ===", flush=True)
    print(cluster_mae, flush=True)
    cluster_mae.to_csv('output/phase22_eda/cluster_mae.csv')
else:
    print("\n  Phase 16 checkpoint not found, skipping MAE per cluster", flush=True)

# ##############################################################
# Q3: rho-band x Position interaction
# ##############################################################
print("\n=== Q3: rho-band x Position Interaction ===", flush=True)

# Position별 평균 rho
print("\nPosition별 평균 rho_max:", flush=True)
print(train.groupby('position_in_scenario')['rho_max'].mean(), flush=True)

# Position별 평균 target
print("\nPosition별 평균 target:", flush=True)
print(train.groupby('position_in_scenario')['avg_delay_minutes_next_30m'].agg(['mean', 'median']), flush=True)

# Cascading 발생 시점 분석
high_rho = train[train['rho_max'] > 0.85]
print(f"\nrho > 0.85 발생 position 분포:", flush=True)
print(high_rho['position_in_scenario'].value_counts().sort_index(), flush=True)
print(f"\n해당 sample target 평균: {high_rho['avg_delay_minutes_next_30m'].mean():.2f}", flush=True)
print(f"전체 sample target 평균: {train['avg_delay_minutes_next_30m'].mean():.2f}", flush=True)

# ##############################################################
# Q4: Bin 9 분석 (Cascading 결과)
# ##############################################################
print("\n=== Q4: Bin 9 (target>100) 특성 ===", flush=True)
bin9 = train[train['avg_delay_minutes_next_30m'] > 100]
non_bin9 = train[train['avg_delay_minutes_next_30m'] <= 100]
print(f"Bin 9: {len(bin9)} samples ({len(bin9)/len(train)*100:.1f}%)", flush=True)

bin9_chars = []
if len(bin9) == 0:
    print("  Bin 9 sample 없음, skip", flush=True)
else:
    print("\nBin 9 평균 vs 일반 평균:", flush=True)
    for f in ['rho_max', 'multi_pressure', 'rho_velocity_3', 'order_inflow_15m',
              'robot_active', 'charge_queue_length', 'congestion_score']:
        if f in train.columns:
            b9 = bin9[f].mean()
            nb = non_bin9[f].mean()
            ratio = b9 / nb if nb != 0 else float('inf')
            print(f"  {f}: Bin9={b9:.2f}, Non-Bin9={nb:.2f}, ratio={ratio:.2f}x", flush=True)
            bin9_chars.append({'feature': f, 'bin9_mean': b9, 'non_bin9_mean': nb, 'ratio': ratio})

    # Bin 9 발생 layout과 position
    print(f"\nBin 9 발생 layout 수: {bin9['layout_id'].nunique()} / {train['layout_id'].nunique()}", flush=True)
    print(f"Bin 9 발생 position 분포:", flush=True)
    print(bin9['position_in_scenario'].value_counts().sort_index(), flush=True)

pd.DataFrame(bin9_chars).to_csv('output/phase22_eda/bin9_characteristics.csv', index=False)

# ##############################################################
# Summary
# ##############################################################
print("\n" + "=" * 60, flush=True)
print("=== Phase 22 EDA Summary ===", flush=True)
print("=" * 60, flush=True)

# 1. Top 3 cascading features
corr_df = pd.DataFrame(corr_results).sort_values('correlation', key=abs, ascending=False)
print(f"\n1. Top 3 cascading features (by |correlation|):", flush=True)
for _, row in corr_df.head(3).iterrows():
    print(f"   {row['feature']}: {row['correlation']:.4f}", flush=True)

# 2. Layout cluster 분리 정도
cluster_std = cluster_target['mean'].std()
print(f"\n2. Layout cluster target mean std: {cluster_std:.2f}", flush=True)

# 3. Test-skewed clusters
n_skewed = (combined_cluster['test_train_ratio'] > 1.5).sum()
print(f"\n3. Test-skewed clusters (ratio > 1.5): {n_skewed}", flush=True)

# 4. rho > 0.85 target ratio
high_rho_mean = high_rho['avg_delay_minutes_next_30m'].mean()
all_mean = train['avg_delay_minutes_next_30m'].mean()
print(f"\n4. rho > 0.85 target mean / all mean: {high_rho_mean:.2f} / {all_mean:.2f} = {high_rho_mean/all_mean:.2f}x", flush=True)

# 5. Bin 9 multi_pressure
if len(bin9) > 0:
    bin9_mp = bin9['multi_pressure'].mean()
    print(f"\n5. Bin 9 multi_pressure mean: {bin9_mp:.2f}", flush=True)
else:
    print(f"\n5. Bin 9 multi_pressure mean: N/A (no samples)", flush=True)

print("\n=== Phase 22 EDA Complete ===", flush=True)
