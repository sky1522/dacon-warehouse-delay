Phase 23 Pre-EDA 스크립트 작성. 우리 데이터 구조 근본적 재검증.

파일: run_phase23_preeda.py
실행 금지, 작성만.
Kaggle CPU 15분 내 실행 가능.

## 배경 (반드시 이해하고 작성)
우리는 22 phases 동안 "25 rows = 시계열"이라고 가정했으나,
같은 scenario 내 shift_hour가 [0,4,0,6,5,6,6,0,6,7,...]로 뒤섞여 있음.
이게 진짜 시계열인지, 독립 snapshot인지 검증이 Day 1 핵심.

5개 EDA로 전체 전략 방향 최종 결정:
- Sequence NN (Track B) 진행 여부
- AMEX 스타일 aggregate features 가치
- Feature removal 후보 식별
- Bin 9 (extreme) 특성
- Distribution shift 원인

## 전체 구조

```python
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

print("="*70)
print("Phase 23 Pre-EDA: 우리 데이터 구조 재검증")
print("="*70)

# Data loading
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
layout = pd.read_csv('data/layout_info.csv')

# Merge layout info
train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

# Output directory
os.makedirs('output/phase23_eda', exist_ok=True)

TARGET = 'avg_delay_minutes_next_30m'

print(f"\nTrain: {train.shape}, Test: {test.shape}, Layout: {layout.shape}")
```

## EDA 1: Sequence Structure 최종 검증 ⭐⭐⭐⭐⭐

**목적:** 25 rows가 시간 순서인가, 독립 snapshot인가 최종 판정.

```python
print("\n" + "="*70)
print("EDA 1: Sequence Structure 검증")
print("="*70)

# 1-1. ID 순서가 의미 있는가?
# 같은 scenario 내 target의 lag autocorrelation
print("\n[1-1] Target Autocorrelation (scenario 내)")

autocorrs = {'lag1': [], 'lag3': [], 'lag5': [], 'lag10': []}
for scn_id in train['scenario_id'].sample(500, random_state=42):
    scn = train[train['scenario_id'] == scn_id].sort_values('ID')
    y = scn[TARGET].values
    for lag in [1, 3, 5, 10]:
        corr = pd.Series(y).autocorr(lag)
        if not np.isnan(corr):
            autocorrs[f'lag{lag}'].append(corr)

autocorr_summary = pd.DataFrame({
    'lag': [1, 3, 5, 10],
    'mean_autocorr': [np.mean(autocorrs[f'lag{l}']) for l in [1,3,5,10]],
    'std_autocorr': [np.std(autocorrs[f'lag{l}']) for l in [1,3,5,10]],
    'median_autocorr': [np.median(autocorrs[f'lag{l}']) for l in [1,3,5,10]],
})
print(autocorr_summary.to_string(index=False))
autocorr_summary.to_csv('output/phase23_eda/01_autocorr.csv', index=False)

# 1-2. Shuffle 검증: scenario 내 rows를 shuffle 해도 통계 유지되는가?
print("\n[1-2] Shuffle Test (scenario 내 row 순서 의미)")
np.random.seed(42)
original_means = []
shuffled_means = []
for scn_id in train['scenario_id'].sample(200, random_state=42):
    scn = train[train['scenario_id'] == scn_id].sort_values('ID')
    y = scn[TARGET].values
    # First half mean vs second half mean (original order)
    orig_first_half = y[:12].mean()
    orig_second_half = y[13:].mean()
    # After shuffle
    shuffled = np.random.permutation(y)
    shuf_first_half = shuffled[:12].mean()
    shuf_second_half = shuffled[13:].mean()
    original_means.append(abs(orig_first_half - orig_second_half))
    shuffled_means.append(abs(shuf_first_half - shuf_second_half))

print(f"  Original |first_half_mean - second_half_mean|: {np.mean(original_means):.3f}")
print(f"  Shuffled |first_half_mean - second_half_mean|: {np.mean(shuffled_means):.3f}")
print(f"  → 차이가 크면 시간 순서 有, 작으면 독립 snapshot")

# 1-3. shift_hour 일관성 (시간 순서면 증가해야 함)
print("\n[1-3] shift_hour Monotonic Check")
monotonic_count = 0
total_count = 0
for scn_id in train['scenario_id'].sample(500, random_state=42):
    scn = train[train['scenario_id'] == scn_id].sort_values('ID')
    hours = scn['shift_hour'].dropna().values
    if len(hours) >= 5:
        # 단조 증가 또는 단조 감소?
        diffs = np.diff(hours)
        if np.all(diffs >= 0) or np.all(diffs <= 0):
            monotonic_count += 1
        total_count += 1

print(f"  Monotonic scenarios: {monotonic_count}/{total_count} ({monotonic_count/total_count*100:.1f}%)")
print(f"  → 50% 미만이면 시간 순서 아님 확정")

# 1-4. 같은 shift_hour 내 target 분산 vs 다른 shift_hour 간 분산
print("\n[1-4] Target Variance by shift_hour")
same_hour_var = train.groupby(['scenario_id', 'shift_hour'])[TARGET].var().mean()
cross_hour_var = train.groupby('scenario_id')[TARGET].var().mean()
print(f"  Same shift_hour variance (within scenario): {same_hour_var:.2f}")
print(f"  Cross shift_hour variance (within scenario): {cross_hour_var:.2f}")
print(f"  Ratio: {same_hour_var/cross_hour_var:.3f}")
print(f"  → 1에 가까우면 shift_hour가 target 설명력 없음")

# 결론 summary
with open('output/phase23_eda/01_conclusion.txt', 'w') as f:
    mean_lag1 = np.mean(autocorrs['lag1'])
    f.write(f"Target lag-1 autocorr: {mean_lag1:.4f}\n")
    f.write(f"Monotonic shift_hour: {monotonic_count/total_count*100:.1f}%\n")
    f.write(f"Shuffled vs original variance diff: {np.mean(original_means):.3f} vs {np.mean(shuffled_means):.3f}\n")
    
    # 판정
    if mean_lag1 > 0.5 and monotonic_count/total_count > 0.5:
        verdict = "TRUE_SEQUENCE"  # Sequence NN 진행
    elif mean_lag1 > 0.3:
        verdict = "WEAK_SEQUENCE"  # Sequence NN 보조
    else:
        verdict = "INDEPENDENT_SNAPSHOTS"  # Sequence NN 스킵
    
    f.write(f"\nVERDICT: {verdict}\n")
    print(f"\n[VERDICT] {verdict}")
```

## EDA 2: Scenario-level Aggregate 가치 ⭐⭐⭐⭐⭐

**목적:** AMEX 스타일 scenario aggregate features가 효과 있는가.

```python
print("\n" + "="*70)
print("EDA 2: Scenario-level Aggregate Feature 가치")
print("="*70)

# 핵심 운영 변수들
key_cols = [
    'order_inflow_15m', 'robot_active', 'robot_utilization',
    'charge_queue_length', 'pack_utilization', 'congestion_score',
    'max_zone_density', 'blocked_path_15m', 'near_collision_15m',
    'fault_count_15m', 'loading_dock_util'
]

# 2-1. Scenario aggregate (mean/std/max/p90)의 target 상관
print("\n[2-1] Scenario Aggregate Correlation with Target")
agg_corrs = []
for col in key_cols:
    if col not in train.columns:
        continue
    # Scenario별 aggregate
    scn_mean = train.groupby('scenario_id')[col].mean()
    scn_std = train.groupby('scenario_id')[col].std()
    scn_max = train.groupby('scenario_id')[col].max()
    scn_p90 = train.groupby('scenario_id')[col].quantile(0.9)
    
    # 각 row에 merge back
    train_tmp = train.copy()
    train_tmp['scn_mean'] = train_tmp['scenario_id'].map(scn_mean)
    train_tmp['scn_std'] = train_tmp['scenario_id'].map(scn_std)
    train_tmp['scn_max'] = train_tmp['scenario_id'].map(scn_max)
    train_tmp['scn_p90'] = train_tmp['scenario_id'].map(scn_p90)
    
    # Target과의 상관
    corrs = {
        'feature': col,
        'row_corr': train_tmp[col].corr(train_tmp[TARGET]),
        'scn_mean_corr': train_tmp['scn_mean'].corr(train_tmp[TARGET]),
        'scn_std_corr': train_tmp['scn_std'].corr(train_tmp[TARGET]),
        'scn_max_corr': train_tmp['scn_max'].corr(train_tmp[TARGET]),
        'scn_p90_corr': train_tmp['scn_p90'].corr(train_tmp[TARGET]),
    }
    agg_corrs.append(corrs)

agg_df = pd.DataFrame(agg_corrs)
print(agg_df.to_string(index=False))
agg_df.to_csv('output/phase23_eda/02_scenario_agg.csv', index=False)

# 2-2. Row value vs scenario-deviation 효과
print("\n[2-2] Scenario Deviation (row - scn_mean) Correlation")
dev_corrs = []
for col in key_cols:
    if col not in train.columns:
        continue
    scn_mean = train.groupby('scenario_id')[col].transform('mean')
    scn_std = train.groupby('scenario_id')[col].transform('std')
    
    deviation = train[col] - scn_mean
    zscore = deviation / (scn_std + 1e-6)
    
    dev_corrs.append({
        'feature': col,
        'raw_corr': train[col].corr(train[TARGET]),
        'deviation_corr': deviation.corr(train[TARGET]),
        'zscore_corr': zscore.corr(train[TARGET])
    })

dev_df = pd.DataFrame(dev_corrs)
print(dev_df.to_string(index=False))
dev_df.to_csv('output/phase23_eda/02_deviation.csv', index=False)

# 2-3. 판정
print("\n[2-3] Verdict")
max_agg_improve = (agg_df[['scn_mean_corr', 'scn_std_corr', 'scn_max_corr', 'scn_p90_corr']].abs().max(axis=1) 
                   - agg_df['row_corr'].abs()).max()
print(f"Max aggregate correlation improvement: {max_agg_improve:.4f}")
print(f"→ 0.05 이상이면 AMEX aggregate 전략 강력, 미만이면 제한적")
```

## EDA 3: 기존 692 Features Importance ⭐⭐⭐⭐

**목적:** Phase 16 ckpt로 feature importance 확인 → removal 후보 식별.

```python
print("\n" + "="*70)
print("EDA 3: 기존 Feature Importance 분석")
print("="*70)

# Phase 16 ckpt 로드 시도
ckpt_path = 'output/ckpt_phase16_lgb_huber.pkl'
if os.path.exists(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    if 'feature_importance' in ckpt:
        imp = ckpt['feature_importance']
        if isinstance(imp, dict):
            imp_df = pd.DataFrame(list(imp.items()), columns=['feature', 'importance'])
        else:
            imp_df = pd.DataFrame({'feature': ckpt.get('feature_names', []), 'importance': imp})
        imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
        imp_df['rank'] = np.arange(len(imp_df)) + 1
        imp_df['cumsum_pct'] = imp_df['importance'].cumsum() / imp_df['importance'].sum() * 100
        
        print(f"\nTotal features: {len(imp_df)}")
        print(f"\nTop 20 features:")
        print(imp_df.head(20).to_string(index=False))
        
        print(f"\nBottom 30 features (removal 후보):")
        print(imp_df.tail(30).to_string(index=False))
        
        # Cumulative analysis
        pct_80 = (imp_df['cumsum_pct'] <= 80).sum()
        pct_95 = (imp_df['cumsum_pct'] <= 95).sum()
        pct_99 = (imp_df['cumsum_pct'] <= 99).sum()
        
        print(f"\nFeatures explaining 80% importance: {pct_80}")
        print(f"Features explaining 95% importance: {pct_95}")
        print(f"Features explaining 99% importance: {pct_99}")
        print(f"Zero importance features: {(imp_df['importance'] == 0).sum()}")
        
        imp_df.to_csv('output/phase23_eda/03_feature_importance.csv', index=False)
        
        # Removal 후보
        remove_candidates = imp_df[imp_df['importance'] == 0]['feature'].tolist()
        with open('output/phase23_eda/03_removal_candidates.txt', 'w') as f:
            f.write(f"Zero importance features ({len(remove_candidates)}):\n")
            f.write('\n'.join(remove_candidates))
    else:
        print("⚠️ 'feature_importance' key not found in ckpt")
else:
    print(f"⚠️ {ckpt_path} not found — Phase 16 ckpt 필요")
    print("이 EDA는 skip. Phase 16 ckpt 복원 후 재실행 권장.")
```

## EDA 4: Bin 9 (Extreme target>100) 특성 ⭐⭐⭐⭐

**목적:** Extreme value 발생 조건 파악.

```python
print("\n" + "="*70)
print("EDA 4: Bin 9 (target>100) 특성 분석")
print("="*70)

bin9 = train[train[TARGET] > 100]
normal = train[train[TARGET] <= 100]

print(f"Bin 9: {len(bin9)} ({len(bin9)/len(train)*100:.2f}%)")
print(f"Normal: {len(normal)}")

# 4-1. shift_hour 분포 차이
print("\n[4-1] shift_hour 분포 (Bin 9 vs Normal)")
bin9_hour = bin9['shift_hour'].value_counts(normalize=True).sort_index()
normal_hour = normal['shift_hour'].value_counts(normalize=True).sort_index()
hour_compare = pd.DataFrame({'bin9_ratio': bin9_hour, 'normal_ratio': normal_hour})
hour_compare['ratio'] = hour_compare['bin9_ratio'] / hour_compare['normal_ratio']
print(hour_compare.to_string())
hour_compare.to_csv('output/phase23_eda/04_bin9_hour.csv')

# 4-2. layout_type 분포
print("\n[4-2] layout_type 분포")
bin9_type = bin9['layout_type'].value_counts(normalize=True)
normal_type = normal['layout_type'].value_counts(normalize=True)
type_compare = pd.DataFrame({'bin9': bin9_type, 'normal': normal_type})
type_compare['ratio'] = type_compare['bin9'] / type_compare['normal']
print(type_compare.to_string())

# 4-3. 주요 변수 차이
print("\n[4-3] Key Feature 차이 (Bin 9 vs Normal)")
bin9_stats = []
for col in key_cols:
    if col not in train.columns:
        continue
    b9_mean = bin9[col].mean()
    nm_mean = normal[col].mean()
    b9_max = bin9[col].max()
    nm_max = normal[col].max()
    ratio = b9_mean / (nm_mean + 1e-6)
    bin9_stats.append({
        'feature': col,
        'bin9_mean': b9_mean,
        'normal_mean': nm_mean,
        'ratio': ratio,
        'bin9_max': b9_max,
        'normal_max': nm_max
    })

bin9_df = pd.DataFrame(bin9_stats).sort_values('ratio', ascending=False)
print(bin9_df.to_string(index=False))
bin9_df.to_csv('output/phase23_eda/04_bin9_features.csv', index=False)

# 4-4. Bin 9 발생 layout 집중도
print("\n[4-4] Bin 9 발생 layout 분포")
b9_layouts = bin9['layout_id'].value_counts()
print(f"Unique Bin 9 layouts: {b9_layouts.nunique()}/250")
print(f"Top 10 layouts with most Bin 9:")
print(b9_layouts.head(10))

# 4-5. Scenario 내 Bin 9 cluster?
print("\n[4-5] Scenario 내 Bin 9 발생 패턴")
scn_bin9 = train.groupby('scenario_id')[TARGET].apply(lambda x: (x > 100).sum())
scn_bin9_dist = scn_bin9.value_counts().sort_index()
print("scenario당 Bin 9 발생 수 분포:")
print(scn_bin9_dist.head(10))
print(f"\n→ scenario당 Bin 9 > 5개면 cluster (cascade처럼)")
print(f"   전부 단발 (<=1)면 random noise")
```

## EDA 5: Distribution Shift 원인 (Adversarial) ⭐⭐⭐⭐

**목적:** Train vs Test 차이 feature 식별.

```python
print("\n" + "="*70)
print("EDA 5: Distribution Shift 분석 (Adversarial)")
print("="*70)

from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# Adversarial labels
train_adv = train.copy()
train_adv['is_test'] = 0
test_adv = test.copy()
test_adv['is_test'] = 1
test_adv[TARGET] = 0  # dummy

combined = pd.concat([train_adv, test_adv], axis=0, ignore_index=True)

# Feature columns (numerical only, exclude IDs and target)
exclude_cols = ['ID', 'layout_id', 'scenario_id', 'layout_type', TARGET, 'is_test']
feature_cols = [c for c in combined.columns if c not in exclude_cols]

# 수치형만
feature_cols = [c for c in feature_cols if combined[c].dtype in [np.float64, np.int64, np.float32]]
print(f"Adversarial features: {len(feature_cols)}")

# GroupKFold by layout (core insight)
X = combined[feature_cols].fillna(-999)
y = combined['is_test']
groups = combined['layout_id']

gkf = GroupKFold(n_splits=5)
adv_importances = []
aucs = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        num_leaves=32, max_depth=6,
        random_state=42, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
    
    preds = model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, preds)
    aucs.append(auc)
    
    imp = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    adv_importances.append(imp)

print(f"\nAdversarial AUC (GroupKFold by layout): {np.mean(aucs):.4f}")
print(f"→ 0.55 미만: train/test 매우 유사, shift 거의 없음")
print(f"→ 0.70 이상: significant shift, 주의 필요")
print(f"→ 0.85 이상: 심각한 shift")

# Top adversarial features (평균 importance)
avg_imp = pd.concat(adv_importances).groupby('feature')['importance'].mean().sort_values(ascending=False)
print(f"\nTop 20 Adversarial Features:")
print(avg_imp.head(20).to_string())

avg_imp.reset_index().to_csv('output/phase23_eda/05_adversarial.csv', index=False)
```

## Summary 생성

```python
print("\n" + "="*70)
print("📋 PHASE 23 EDA SUMMARY")
print("="*70)

summary = {
    'sequence_verdict': 'TBD',  # EDA 1에서 채움
    'aggregate_value': 'TBD',   # EDA 2에서 채움
    'feature_removal_candidates': 0,  # EDA 3
    'bin9_key_features': [],    # EDA 4
    'adversarial_auc': 0.0      # EDA 5
}

# 파일로 저장
with open('output/phase23_eda/SUMMARY.txt', 'w') as f:
    f.write("="*50 + "\n")
    f.write("Phase 23 EDA Summary\n")
    f.write("="*50 + "\n\n")
    
    f.write("EDA 1: Sequence Structure\n")
    f.write(f"  Target lag-1 autocorr: {np.mean(autocorrs['lag1']):.4f}\n")
    f.write(f"  Monotonic shift_hour: {monotonic_count/total_count*100:.1f}%\n")
    f.write(f"  Verdict: {'TRUE_SEQUENCE' if np.mean(autocorrs['lag1']) > 0.5 else 'INDEPENDENT_SNAPSHOTS' if np.mean(autocorrs['lag1']) < 0.3 else 'WEAK_SEQUENCE'}\n\n")
    
    f.write("EDA 2: Scenario Aggregate\n")
    f.write(f"  Max improvement over row-level: {max_agg_improve:.4f}\n")
    f.write(f"  Verdict: {'STRONG' if max_agg_improve > 0.05 else 'WEAK'}\n\n")
    
    f.write("EDA 3: Feature Importance\n")
    if os.path.exists('output/phase23_eda/03_feature_importance.csv'):
        f.write(f"  Total features: {len(imp_df)}\n")
        f.write(f"  Zero importance: {(imp_df['importance'] == 0).sum()}\n")
        f.write(f"  80% explanation: {(imp_df['cumsum_pct'] <= 80).sum()} features\n")
    
    f.write("\nEDA 4: Bin 9 Characteristics\n")
    f.write(f"  Bin 9 count: {len(bin9)} ({len(bin9)/len(train)*100:.2f}%)\n")
    f.write(f"  Top feature ratio: {bin9_df.iloc[0]['feature']} = {bin9_df.iloc[0]['ratio']:.2f}x\n")
    
    f.write(f"\nEDA 5: Adversarial AUC = {np.mean(aucs):.4f}\n")

print("\n✅ All EDA complete. Results in output/phase23_eda/")
print("\n🎯 Next: Review SUMMARY.txt → Phase 23 Track A/B/C 결정")
```

## 작성 완료 체크

- 실행 금지, 작성만
- output/phase23_eda/ 디렉토리 생성 확인
- 5개 EDA의 csv 파일과 SUMMARY.txt 저장
- ast.parse 통과
- 커밋 메시지: "feat: Phase 23 Pre-EDA - 5 critical structure checks"
- 푸시