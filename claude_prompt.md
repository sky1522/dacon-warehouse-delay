run_phase22_eda.py 2가지 critical 버그 수정.

## Bug 1: Layout 컬럼 merge 누락
data/train.csv에 robot_total, aisle_width_avg 등 layout 컬럼이 없음.
data/layout_info.csv를 layout_id로 merge 필요.

수정:
```python
import pandas as pd, numpy as np
import os

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout_info = pd.read_csv('data/layout_info.csv')

print(f"Train cols: {train.columns.tolist()[:10]}...")
print(f"Layout cols: {layout_info.columns.tolist()}")

# Merge layout features
train = train.merge(layout_info, on='layout_id', how='left')
test = test.merge(layout_info, on='layout_id', how='left')

print(f"After merge: train shape {train.shape}, test shape {test.shape}")
```

## Bug 2: position_in_scenario 컬럼 없음
scenario 내 timestep 순서는 cumcount로 생성:

```python
# Position 생성 (scenario_id 내 순서)
train = train.sort_values(['scenario_id'])  # 또는 ID 순서 유지
train['position_in_scenario'] = train.groupby('scenario_id').cumcount()
test['position_in_scenario'] = test.groupby('scenario_id').cumcount()
```

만약 train.csv에 timestep을 알 수 있는 다른 컬럼(예: ID, time 등) 있으면 그걸 우선 사용.

## Bug 3: NaN 명시 처리 (rho_diff, rho_velocity_3)
```python
# NaN을 0으로 채워서 명시 (correlation 영향 X, 통계는 정확)
train['rho_diff'] = train['rho_diff'].fillna(0)
train['rho_velocity_3'] = train['rho_velocity_3'].fillna(0)
```

## Bug 4: Bin 9 empty guard
```python
if len(bin9) == 0:
    print("⚠️ Bin 9 sample 없음, skip")
else:
    # 기존 로직
```

## Bug 5: KMeans는 layout_info에서 직접 시작
train/test 거치지 말고 layout_info에서 바로 cluster:

```python
layout_features = ['aisle_width_avg', 'robot_total', 'intersection_count',
                    'layout_compactness', 'pack_station_count', 'charger_count',
                    'one_way_ratio', 'floor_area_sqm', 'zone_dispersion']

# layout_info에서 직접 KMeans
X_layouts = layout_info[layout_features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_layouts)

km = KMeans(n_clusters=10, random_state=42, n_init=10)
layout_info['cluster'] = km.fit_predict(X_scaled)

# Train/Test에 cluster 할당
train_layout_ids = train['layout_id'].unique()
test_layout_ids = test['layout_id'].unique()

train_clusters = layout_info[layout_info['layout_id'].isin(train_layout_ids)]['cluster']
test_clusters = layout_info[layout_info['layout_id'].isin(test_layout_ids)]['cluster']

print(f"Train layout clusters: {train_clusters.value_counts().sort_index().to_dict()}")
print(f"Test layout clusters: {test_clusters.value_counts().sort_index().to_dict()}")
```

## 작업 후
- 데이터 컬럼 확인 print 추가 (디버깅용)
- 수정 후 ast.parse 확인
- 커밋: "fix: Phase 22 EDA - layout merge + position cumcount + NaN handling"
- 푸시