# Phase 22 Pre-EDA: Cascading + Layout Cluster Validation

## 스크립트
- `run_phase22_eda.py` (Kaggle 30분 내 실행 가능)

## 4가지 검증
1. **Q1: Cascading Detector** - rho utilization, binary indicators, multi_pressure, rho velocity → target correlation
2. **Q2: Layout Clustering** - K-means 10 cluster, train/test 분포 비교, cluster별 target/MAE
3. **Q3: rho-band x Position** - position별 rho/target 분포, cascading 발생 시점
4. **Q4: Bin 9 특성** - target>100 sample의 rho/pressure 특성

## 출력 파일
- `output/phase22_eda/cascading_correlations.csv`
- `output/phase22_eda/layout_clusters.csv`
- `output/phase22_eda/cluster_mae.csv`
- `output/phase22_eda/bin9_characteristics.csv`
