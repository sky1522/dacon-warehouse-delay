# Phase 22: Cascading Binary + Layout Cluster Features

## 스크립트
- `run_phase22_cascade_cluster.py` (Phase 16 base + 12 new features)

## 추가 Features (12개)
### A. Cascading Binary (5)
- `rho_over_70`, `rho_over_85`, `rho_over_95`: rho_max 임계값 indicators
- `multi_pressure`: robot+pack+charger 동시 과부하 count
- `explosion_intensity`: rho_max * multi_pressure

### B. Layout Cluster (4)
- KMeans 10 cluster on layout_info
- `cluster_mean_target`, `cluster_p95_target`, `cluster_bin9_rate`, `cluster_size`
- Train-only 통계 (test leakage 방지)

### C. Cross Features (3)
- `rho85_x_clusterbin9`: rho>0.85 * cluster bin9 rate
- `multipress_x_compact`: multi_pressure * layout_compactness
- `explosion_x_narrow`: explosion_intensity * (aisle_width < 2.5)

## 판단 기준
- Ensemble CV < 8.4403 → 새 features 효과 있음
- Phase 22 new features importance rank 확인
