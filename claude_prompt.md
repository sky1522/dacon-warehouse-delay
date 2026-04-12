Phase 22 본 작업: Cascading + Layout Cluster Features 추가

Base: run_phase16_fe.py 복사 → run_phase22_cascade_cluster.py
실행 금지, 작성만.

## 추가 Features (Phase 17 FE 뒤에)

### A. Cascading Binary (5개)
```python
# NaN 처리된 후, FE 뒤에 추가
combined['rho_robot'] = combined['robot_active'] / (combined['robot_total'] + 1e-6)
combined['rho_pack'] = combined['pack_utilization'].fillna(0.3)
combined['rho_charger'] = combined['charge_queue_length'] / (combined['charger_count'] + 1e-6)
combined['rho_max_new'] = combined[['rho_robot', 'rho_pack', 'rho_charger']].max(axis=1)

combined['rho_over_70'] = (combined['rho_max_new'] > 0.70).astype(int)
combined['rho_over_85'] = (combined['rho_max_new'] > 0.85).astype(int)
combined['rho_over_95'] = (combined['rho_max_new'] > 0.95).astype(int)

combined['robot_pressure'] = (combined['rho_robot'] > 0.85).astype(int)
combined['pack_pressure'] = (combined['pack_utilization'] > 0.80).astype(int)
combined['charger_pressure'] = (combined['charge_queue_length'] > 5).astype(int)
combined['multi_pressure'] = combined['robot_pressure'] + combined['pack_pressure'] + combined['charger_pressure']

combined['explosion_intensity'] = combined['rho_max_new'] * combined['multi_pressure']
```

### B. Layout Cluster (4개)
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

layout_info = pd.read_csv('data/layout_info.csv')
layout_features = ['aisle_width_avg', 'robot_total', 'intersection_count',
                    'layout_compactness', 'pack_station_count', 'charger_count',
                    'one_way_ratio', 'floor_area_sqm', 'zone_dispersion']

X = layout_info[layout_features].fillna(0).values
X_scaled = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=10, random_state=42, n_init=10)
layout_info['layout_cluster'] = km.fit_predict(X_scaled)

# Cluster별 통계 (train만 사용, test leakage 방지)
cluster_stats = train_df.merge(
    layout_info[['layout_id', 'layout_cluster']], on='layout_id'
).groupby('layout_cluster').agg(
    cluster_mean_target=('avg_delay_minutes_next_30m', 'mean'),
    cluster_p95_target=('avg_delay_minutes_next_30m', lambda x: x.quantile(0.95)),
    cluster_bin9_rate=('avg_delay_minutes_next_30m', lambda x: (x > 100).mean()),
    cluster_size=('avg_delay_minutes_next_30m', 'count')
).reset_index()

# combined에 merge
combined = combined.merge(layout_info[['layout_id', 'layout_cluster']], on='layout_id', how='left')
combined = combined.merge(cluster_stats, on='layout_cluster', how='left')
```

### C. Cross Features (3개)
```python
combined['rho85_x_clusterbin9'] = combined['rho_over_85'] * combined['cluster_bin9_rate']
combined['multipress_x_compact'] = combined['multi_pressure'] * combined['layout_compactness']
combined['explosion_x_narrow'] = combined['explosion_intensity'] * (combined['aisle_width_avg'] < 2.5).astype(int)
```

## 모델
- Phase 16과 동일한 7 models (LGB×3 + XGB + Cat×2 + MLP)
- TabNet 포함 (Kaggle 환경에서 pytorch-tabnet OK)
- 총 feature: 692 + 12 = 704
- 체크포인트: ckpt_phase22_*.pkl
- Submission: submission_phase22.csv

## 결과 출력
- 7 모델 CV (Phase 16 대비)
- Ensemble CV
- 새 12 features의 importance 순위
- Cluster별 MAE

커밋: feat: Phase 22 - cascading binary + layout cluster features
푸시.