import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70, flush=True)
print("Phase 23 Pre-EDA: 우리 데이터 구조 재검증", flush=True)
print("=" * 70, flush=True)

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

print(f"\nTrain: {train.shape}, Test: {test.shape}, Layout: {layout.shape}", flush=True)

# ##############################################################
# EDA 1: Sequence Structure 최종 검증
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("EDA 1: Sequence Structure 검증", flush=True)
print("=" * 70, flush=True)

# 1-1. Target Autocorrelation (scenario 내)
print("\n[1-1] Target Autocorrelation (scenario 내)", flush=True)

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
    'mean_autocorr': [np.mean(autocorrs[f'lag{l}']) for l in [1, 3, 5, 10]],
    'std_autocorr': [np.std(autocorrs[f'lag{l}']) for l in [1, 3, 5, 10]],
    'median_autocorr': [np.median(autocorrs[f'lag{l}']) for l in [1, 3, 5, 10]],
})
print(autocorr_summary.to_string(index=False), flush=True)
autocorr_summary.to_csv('output/phase23_eda/01_autocorr.csv', index=False)

# 1-2. Shuffle 검증
print("\n[1-2] Shuffle Test (scenario 내 row 순서 의미)", flush=True)
np.random.seed(42)
original_means = []
shuffled_means = []
for scn_id in train['scenario_id'].sample(200, random_state=42):
    scn = train[train['scenario_id'] == scn_id].sort_values('ID')
    y = scn[TARGET].values
    orig_first_half = y[:12].mean()
    orig_second_half = y[13:].mean()
    shuffled = np.random.permutation(y)
    shuf_first_half = shuffled[:12].mean()
    shuf_second_half = shuffled[13:].mean()
    original_means.append(abs(orig_first_half - orig_second_half))
    shuffled_means.append(abs(shuf_first_half - shuf_second_half))

print(f"  Original |first_half_mean - second_half_mean|: {np.mean(original_means):.3f}", flush=True)
print(f"  Shuffled |first_half_mean - second_half_mean|: {np.mean(shuffled_means):.3f}", flush=True)
print(f"  -> 차이가 크면 시간 순서 有, 작으면 독립 snapshot", flush=True)

# 1-3. shift_hour Monotonic Check
print("\n[1-3] shift_hour Monotonic Check", flush=True)
monotonic_count = 0
total_count = 0
for scn_id in train['scenario_id'].sample(500, random_state=42):
    scn = train[train['scenario_id'] == scn_id].sort_values('ID')
    hours = scn['shift_hour'].dropna().values
    if len(hours) >= 5:
        diffs = np.diff(hours)
        if np.all(diffs >= 0) or np.all(diffs <= 0):
            monotonic_count += 1
        total_count += 1

print(f"  Monotonic scenarios: {monotonic_count}/{total_count} ({monotonic_count/total_count*100:.1f}%)", flush=True)
print(f"  -> 50% 미만이면 시간 순서 아님 확정", flush=True)

# 1-4. Target Variance by shift_hour
print("\n[1-4] Target Variance by shift_hour", flush=True)
same_hour_var = train.groupby(['scenario_id', 'shift_hour'])[TARGET].var().mean()
cross_hour_var = train.groupby('scenario_id')[TARGET].var().mean()
print(f"  Same shift_hour variance (within scenario): {same_hour_var:.2f}", flush=True)
print(f"  Cross shift_hour variance (within scenario): {cross_hour_var:.2f}", flush=True)
print(f"  Ratio: {same_hour_var / cross_hour_var:.3f}", flush=True)
print(f"  -> 1에 가까우면 shift_hour가 target 설명력 없음", flush=True)

# EDA 1 결론
mean_lag1 = np.mean(autocorrs['lag1'])
mono_pct = monotonic_count / total_count if total_count > 0 else 0

if mean_lag1 > 0.5 and mono_pct > 0.5:
    verdict_1 = "TRUE_SEQUENCE"
elif mean_lag1 > 0.3:
    verdict_1 = "WEAK_SEQUENCE"
else:
    verdict_1 = "INDEPENDENT_SNAPSHOTS"

with open('output/phase23_eda/01_conclusion.txt', 'w') as f:
    f.write(f"Target lag-1 autocorr: {mean_lag1:.4f}\n")
    f.write(f"Monotonic shift_hour: {mono_pct * 100:.1f}%\n")
    f.write(f"Shuffled vs original variance diff: {np.mean(original_means):.3f} vs {np.mean(shuffled_means):.3f}\n")
    f.write(f"\nVERDICT: {verdict_1}\n")

print(f"\n[VERDICT] {verdict_1}", flush=True)

# ##############################################################
# EDA 2: Scenario-level Aggregate 가치
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("EDA 2: Scenario-level Aggregate Feature 가치", flush=True)
print("=" * 70, flush=True)

key_cols = [
    'order_inflow_15m', 'robot_active', 'robot_utilization',
    'charge_queue_length', 'pack_utilization', 'congestion_score',
    'max_zone_density', 'blocked_path_15m', 'near_collision_15m',
    'fault_count_15m', 'loading_dock_util'
]

# 2-1. Scenario aggregate correlation
print("\n[2-1] Scenario Aggregate Correlation with Target", flush=True)
agg_corrs = []
for col in key_cols:
    if col not in train.columns:
        continue
    scn_mean = train.groupby('scenario_id')[col].mean()
    scn_std = train.groupby('scenario_id')[col].std()
    scn_max = train.groupby('scenario_id')[col].max()
    scn_p90 = train.groupby('scenario_id')[col].quantile(0.9)

    train_tmp = train.copy()
    train_tmp['scn_mean'] = train_tmp['scenario_id'].map(scn_mean)
    train_tmp['scn_std'] = train_tmp['scenario_id'].map(scn_std)
    train_tmp['scn_max'] = train_tmp['scenario_id'].map(scn_max)
    train_tmp['scn_p90'] = train_tmp['scenario_id'].map(scn_p90)

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
print(agg_df.to_string(index=False), flush=True)
agg_df.to_csv('output/phase23_eda/02_scenario_agg.csv', index=False)

# 2-2. Scenario deviation correlation
print("\n[2-2] Scenario Deviation (row - scn_mean) Correlation", flush=True)
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
print(dev_df.to_string(index=False), flush=True)
dev_df.to_csv('output/phase23_eda/02_deviation.csv', index=False)

# 2-3. Verdict
print("\n[2-3] Verdict", flush=True)
max_agg_improve = (agg_df[['scn_mean_corr', 'scn_std_corr', 'scn_max_corr', 'scn_p90_corr']].abs().max(axis=1)
                   - agg_df['row_corr'].abs()).max()
print(f"Max aggregate correlation improvement: {max_agg_improve:.4f}", flush=True)
print(f"-> 0.05 이상이면 AMEX aggregate 전략 강력, 미만이면 제한적", flush=True)

# ##############################################################
# EDA 3: 기존 692 Features Importance
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("EDA 3: 기존 Feature Importance 분석", flush=True)
print("=" * 70, flush=True)

imp_df = None
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

        print(f"\nTotal features: {len(imp_df)}", flush=True)
        print(f"\nTop 20 features:", flush=True)
        print(imp_df.head(20).to_string(index=False), flush=True)

        print(f"\nBottom 30 features (removal 후보):", flush=True)
        print(imp_df.tail(30).to_string(index=False), flush=True)

        pct_80 = (imp_df['cumsum_pct'] <= 80).sum()
        pct_95 = (imp_df['cumsum_pct'] <= 95).sum()
        pct_99 = (imp_df['cumsum_pct'] <= 99).sum()

        print(f"\nFeatures explaining 80% importance: {pct_80}", flush=True)
        print(f"Features explaining 95% importance: {pct_95}", flush=True)
        print(f"Features explaining 99% importance: {pct_99}", flush=True)
        print(f"Zero importance features: {(imp_df['importance'] == 0).sum()}", flush=True)

        imp_df.to_csv('output/phase23_eda/03_feature_importance.csv', index=False)

        remove_candidates = imp_df[imp_df['importance'] == 0]['feature'].tolist()
        with open('output/phase23_eda/03_removal_candidates.txt', 'w') as f:
            f.write(f"Zero importance features ({len(remove_candidates)}):\n")
            f.write('\n'.join(remove_candidates))
    else:
        print("  'feature_importance' key not found in ckpt", flush=True)
else:
    print(f"  {ckpt_path} not found -- Phase 16 ckpt 필요", flush=True)
    print("  이 EDA는 skip. Phase 16 ckpt 복원 후 재실행 권장.", flush=True)

# ##############################################################
# EDA 4: Bin 9 (Extreme target>100) 특성
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("EDA 4: Bin 9 (target>100) 특성 분석", flush=True)
print("=" * 70, flush=True)

bin9 = train[train[TARGET] > 100]
normal = train[train[TARGET] <= 100]

print(f"Bin 9: {len(bin9)} ({len(bin9) / len(train) * 100:.2f}%)", flush=True)
print(f"Normal: {len(normal)}", flush=True)

if len(bin9) == 0:
    print("  Bin 9 sample 없음, skip", flush=True)
    bin9_df = pd.DataFrame()
else:
    # 4-1. shift_hour 분포
    print("\n[4-1] shift_hour 분포 (Bin 9 vs Normal)", flush=True)
    bin9_hour = bin9['shift_hour'].value_counts(normalize=True).sort_index()
    normal_hour = normal['shift_hour'].value_counts(normalize=True).sort_index()
    hour_compare = pd.DataFrame({'bin9_ratio': bin9_hour, 'normal_ratio': normal_hour})
    hour_compare['ratio'] = hour_compare['bin9_ratio'] / hour_compare['normal_ratio']
    print(hour_compare.to_string(), flush=True)
    hour_compare.to_csv('output/phase23_eda/04_bin9_hour.csv')

    # 4-2. layout_type 분포
    print("\n[4-2] layout_type 분포", flush=True)
    bin9_type = bin9['layout_type'].value_counts(normalize=True)
    normal_type = normal['layout_type'].value_counts(normalize=True)
    type_compare = pd.DataFrame({'bin9': bin9_type, 'normal': normal_type})
    type_compare['ratio'] = type_compare['bin9'] / type_compare['normal']
    print(type_compare.to_string(), flush=True)

    # 4-3. Key feature 차이
    print("\n[4-3] Key Feature 차이 (Bin 9 vs Normal)", flush=True)
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
    print(bin9_df.to_string(index=False), flush=True)
    bin9_df.to_csv('output/phase23_eda/04_bin9_features.csv', index=False)

    # 4-4. Bin 9 발생 layout 집중도
    print("\n[4-4] Bin 9 발생 layout 분포", flush=True)
    b9_layouts = bin9['layout_id'].value_counts()
    print(f"Unique Bin 9 layouts: {b9_layouts.nunique()}/250", flush=True)
    print(f"Top 10 layouts with most Bin 9:", flush=True)
    print(b9_layouts.head(10), flush=True)

    # 4-5. Scenario 내 Bin 9 cluster
    print("\n[4-5] Scenario 내 Bin 9 발생 패턴", flush=True)
    scn_bin9 = train.groupby('scenario_id')[TARGET].apply(lambda x: (x > 100).sum())
    scn_bin9_dist = scn_bin9.value_counts().sort_index()
    print("scenario당 Bin 9 발생 수 분포:", flush=True)
    print(scn_bin9_dist.head(10), flush=True)
    print(f"\n-> scenario당 Bin 9 > 5개면 cluster (cascade처럼)", flush=True)
    print(f"   전부 단발 (<=1)면 random noise", flush=True)

# ##############################################################
# EDA 5: Distribution Shift 원인 (Adversarial)
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("EDA 5: Distribution Shift 분석 (Adversarial)", flush=True)
print("=" * 70, flush=True)

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
feature_cols = [c for c in feature_cols if combined[c].dtype in [np.float64, np.int64, np.float32]]
print(f"Adversarial features: {len(feature_cols)}", flush=True)

# GroupKFold by layout
X_adv = combined[feature_cols].fillna(-999)
y_adv = combined['is_test']
groups_adv = combined['layout_id']

gkf = GroupKFold(n_splits=5)
adv_importances = []
aucs = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_adv, y_adv, groups_adv)):
    X_tr, X_val = X_adv.iloc[tr_idx], X_adv.iloc[val_idx]
    y_tr, y_val = y_adv.iloc[tr_idx], y_adv.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        num_leaves=32, max_depth=6,
        random_state=42, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    aucs.append(auc)

    imp_fold = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    adv_importances.append(imp_fold)

print(f"\nAdversarial AUC (GroupKFold by layout): {np.mean(aucs):.4f}", flush=True)
print(f"-> 0.55 미만: train/test 매우 유사, shift 거의 없음", flush=True)
print(f"-> 0.70 이상: significant shift, 주의 필요", flush=True)
print(f"-> 0.85 이상: 심각한 shift", flush=True)

# Top adversarial features
avg_imp = pd.concat(adv_importances).groupby('feature')['importance'].mean().sort_values(ascending=False)
print(f"\nTop 20 Adversarial Features:", flush=True)
print(avg_imp.head(20).to_string(), flush=True)

avg_imp.reset_index().to_csv('output/phase23_eda/05_adversarial.csv', index=False)

# ##############################################################
# Summary
# ##############################################################
print("\n" + "=" * 70, flush=True)
print("PHASE 23 EDA SUMMARY", flush=True)
print("=" * 70, flush=True)

with open('output/phase23_eda/SUMMARY.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("Phase 23 EDA Summary\n")
    f.write("=" * 50 + "\n\n")

    f.write("EDA 1: Sequence Structure\n")
    f.write(f"  Target lag-1 autocorr: {mean_lag1:.4f}\n")
    f.write(f"  Monotonic shift_hour: {mono_pct * 100:.1f}%\n")
    f.write(f"  Verdict: {verdict_1}\n\n")

    f.write("EDA 2: Scenario Aggregate\n")
    f.write(f"  Max improvement over row-level: {max_agg_improve:.4f}\n")
    verdict_2 = 'STRONG' if max_agg_improve > 0.05 else 'WEAK'
    f.write(f"  Verdict: {verdict_2}\n\n")

    f.write("EDA 3: Feature Importance\n")
    if imp_df is not None:
        f.write(f"  Total features: {len(imp_df)}\n")
        f.write(f"  Zero importance: {(imp_df['importance'] == 0).sum()}\n")
        f.write(f"  80% explanation: {(imp_df['cumsum_pct'] <= 80).sum()} features\n")
    else:
        f.write("  Skipped (no ckpt)\n")

    f.write("\nEDA 4: Bin 9 Characteristics\n")
    f.write(f"  Bin 9 count: {len(bin9)} ({len(bin9) / len(train) * 100:.2f}%)\n")
    if len(bin9_df) > 0:
        f.write(f"  Top feature ratio: {bin9_df.iloc[0]['feature']} = {bin9_df.iloc[0]['ratio']:.2f}x\n")

    f.write(f"\nEDA 5: Adversarial AUC = {np.mean(aucs):.4f}\n")

print("\nAll EDA complete. Results in output/phase23_eda/", flush=True)
print("\nNext: Review SUMMARY.txt -> Phase 23 Track A/B/C 결정", flush=True)
