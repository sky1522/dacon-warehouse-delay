Phase 22 Pre-EDA 스크립트 작성. 도메인 기반 가설 검증.

파일: run_phase22_eda.py (Kaggle에서 30분 내 실행)
실행 금지, 작성만.

## 목적
3가지 도메인 가설을 데이터로 검증:
1. Cascading detector features가 target과 correlation 있는가
2. Layout cluster가 의미 있게 분리되는가  
3. ρ-band별 target 분포가 다른가

## 작업

### Q1: Cascading Detector Validation

```python
import pandas as pd, numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 1. ρ utilization 계산
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

# 4. ρ velocity (scenario 내 변화율)
train = train.sort_values(['scenario_id', 'position_in_scenario'])
train['rho_diff'] = train.groupby('scenario_id')['rho_max'].diff()
train['rho_velocity_3'] = train.groupby('scenario_id')['rho_max'].rolling(3).apply(
    lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 3 else 0
).reset_index(level=0, drop=True)

# 5. Target과 correlation 측정
target = train['avg_delay_minutes_next_30m']

print("=== Q1: Cascading Detector Correlation ===")
cascading_features = ['rho_max', 'rho_over_70', 'rho_over_85', 'rho_over_95',
                       'multi_pressure', 'rho_diff', 'rho_velocity_3']
for f in cascading_features:
    corr = train[f].corr(target)
    print(f"  {f}: corr = {corr:.4f}")

# 6. ρ-band별 target 통계
print("\n=== Q1-2: Target by ρ-band ===")
bands = pd.cut(train['rho_max'], bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0])
print(train.groupby(bands)['avg_delay_minutes_next_30m'].agg(['count', 'mean', 'median', 'max', 'std']))
```

### Q2: Layout Cluster Validation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("\n=== Q2: Layout Clustering ===")

# Layout-level features (각 layout 1번씩 추출)
layout_features = ['aisle_width_avg', 'robot_total', 'intersection_count',
                    'layout_compactness', 'pack_station_count', 'charger_count',
                    'one_way_ratio', 'floor_area_sqm', 'zone_dispersion']

# train + test layout 합치기 (350 unique)
all_layouts_train = train.groupby('layout_id')[layout_features].first()
all_layouts_test = test.groupby('layout_id')[layout_features].first()
all_layouts = pd.concat([all_layouts_train, all_layouts_test]).drop_duplicates()
print(f"Total unique layouts: {len(all_layouts)}")

# K-means 10개 cluster
scaler = StandardScaler()
X_layouts = scaler.fit_transform(all_layouts)
km = KMeans(n_clusters=10, random_state=42, n_init=10)
all_layouts['cluster'] = km.fit_predict(X_layouts)

# Train/Test layout이 어느 cluster에 속하는지
train_layout_cluster = all_layouts.loc[all_layouts_train.index, 'cluster']
test_layout_cluster = all_layouts.loc[all_layouts_test.index, 'cluster']

print("\nCluster 분포:")
print(f"  Train layouts (250): {train_layout_cluster.value_counts().sort_index().tolist()}")
print(f"  Test layouts (100):  {test_layout_cluster.value_counts().sort_index().tolist()}")

# Test에만 많이 있는 cluster?
combined = pd.DataFrame({
    'train': train_layout_cluster.value_counts().sort_index(),
    'test': test_layout_cluster.value_counts().sort_index()
}).fillna(0)
combined['test_train_ratio'] = combined['test'] / (combined['train'] + 1)
print("\nTest/Train ratio per cluster:")
print(combined.sort_values('test_train_ratio', ascending=False))

# Cluster별 target 분포 (train만)
train['layout_cluster'] = train['layout_id'].map(train_layout_cluster)
print("\n=== Cluster별 Target 통계 ===")
print(train.groupby('layout_cluster')['avg_delay_minutes_next_30m'].agg(['count', 'mean', 'median', 'std']))

# Phase 16 OOF MAE per cluster (있으면)
import os
if os.path.exists('output/ckpt_phase16_lgb_huber.pkl'):
    import pickle
    with open('output/ckpt_phase16_lgb_huber.pkl', 'rb') as f:
        ckpt = pickle.load(f)
    train['phase16_oof'] = ckpt['oof']
    train['mae'] = np.abs(train['phase16_oof'] - train['avg_delay_minutes_next_30m'])
    print("\n=== Cluster별 Phase 16 MAE ===")
    print(train.groupby('layout_cluster')['mae'].agg(['count', 'mean', 'median']).sort_values('mean', ascending=False))
```

### Q3: ρ-band × Position interaction

```python
print("\n=== Q3: ρ-band × Position Interaction ===")

# Position별 평균 ρ
print("\nPosition별 평균 ρ_max:")
print(train.groupby('position_in_scenario')['rho_max'].mean())

# Position별 평균 target
print("\nPosition별 평균 target:")
print(train.groupby('position_in_scenario')['avg_delay_minutes_next_30m'].agg(['mean', 'median']))

# Cascading 발생 시점 분석
high_rho = train[train['rho_max'] > 0.85]
print(f"\nρ > 0.85 발생 position 분포:")
print(high_rho['position_in_scenario'].value_counts().sort_index())
print(f"\n해당 sample target 평균: {high_rho['avg_delay_minutes_next_30m'].mean():.2f}")
print(f"전체 sample target 평균: {train['avg_delay_minutes_next_30m'].mean():.2f}")
```

### Q4: Bin 9 분석 (Cascading 결과)

```python
print("\n=== Q4: Bin 9 (target>100) 특성 ===")
bin9 = train[train['avg_delay_minutes_next_30m'] > 100]
non_bin9 = train[train['avg_delay_minutes_next_30m'] <= 100]
print(f"Bin 9: {len(bin9)} samples ({len(bin9)/len(train)*100:.1f}%)")

print("\nBin 9 평균 vs 일반 평균:")
for f in ['rho_max', 'multi_pressure', 'rho_velocity_3', 'order_inflow_15m',
          'robot_active', 'charge_queue_length', 'congestion_score']:
    if f in train.columns:
        b9 = bin9[f].mean()
        nb = non_bin9[f].mean()
        print(f"  {f}: Bin9={b9:.2f}, Non-Bin9={nb:.2f}, ratio={b9/nb:.2f}x")

# Bin 9 발생 layout과 position
print(f"\nBin 9 발생 layout 수: {bin9['layout_id'].nunique()} / 250")
print(f"Bin 9 발생 position 분포:")
print(bin9['position_in_scenario'].value_counts().sort_index())
```

## 출력 파일
- output/phase22_eda/cascading_correlations.csv
- output/phase22_eda/layout_clusters.csv (cluster 정보)
- output/phase22_eda/cluster_mae.csv (cluster별 MAE)
- output/phase22_eda/bin9_characteristics.csv

## 결과 요약 출력
스크립트 마지막에 핵심 5개:
1. Top 3 cascading features (correlation 가장 큰 것)
2. Layout cluster 분리 정도 (target std 차이)
3. Test-skewed cluster 개수 (test/train ratio > 1.5)
4. ρ > 0.85 sample의 target 평균 vs 전체 평균 비율
5. Bin 9 sample의 multi_pressure 평균

규칙:
- 작성만, 실행 금지
- 커밋: feat: Phase 22 Pre-EDA - cascading + layout cluster validation
- 푸시