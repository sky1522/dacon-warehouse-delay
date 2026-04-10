"""
Phase 13 Step 4: Bin 9 (extreme target > 100) Characterization
- KS test: which features distinguish bin 9?
- Clustering: are there sub-patterns within bin 9?
- Condition search: what feature combinations trigger bin 9?
- Classifier: can we predict bin 9 occurrence? (AUC)
- Test set: does test have more bin 9 candidates?
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output/phase13s4_bin9', exist_ok=True)

# ============================================================
# Section 1: Bin 9 definition + basic stats
# ============================================================
print("=" * 60)
print("Section 1: Bin 9 definition + basic stats")
print("=" * 60)

train_df = pd.read_csv('data/train.csv')
layout_info = pd.read_csv('data/layout_info.csv')
train_df = train_df.merge(layout_info, on='layout_id', how='left')

with open('output/ckpt_phase13s1_lgb_huber.pkl', 'rb') as f:
    ckpt = pickle.load(f)
train_df['oof'] = ckpt['oof']
train_df['residual'] = train_df['avg_delay_minutes_next_30m'] - train_df['oof']

y = train_df['avg_delay_minutes_next_30m']
THRESHOLD = 100
train_df['is_bin9'] = (y > THRESHOLD).astype(int)

n_bin9 = train_df['is_bin9'].sum()
print(f"Threshold: target > {THRESHOLD}")
print(f"Bin 9 samples: {n_bin9} ({n_bin9/len(train_df)*100:.2f}%)")
print(f"Bin 9 target: mean={y[y>THRESHOLD].mean():.1f}, median={y[y>THRESHOLD].median():.1f}, max={y.max():.1f}")
print(f"Non-bin9 target: mean={y[y<=THRESHOLD].mean():.1f}")

bin9_oof_mean = train_df.loc[train_df['is_bin9'] == 1, 'oof'].mean()
bin9_residual_mean = train_df.loc[train_df['is_bin9'] == 1, 'residual'].mean()
print(f"\nBin 9 OOF prediction: mean={bin9_oof_mean:.1f}")
print(f"Bin 9 residual: mean={bin9_residual_mean:.1f}")
print(f"  -> Model predicts avg {bin9_oof_mean:.1f} but actual avg is {y[y>THRESHOLD].mean():.1f}")

# ============================================================
# Section 2: Bin 9 vs non-bin9 KS test (all features)
# ============================================================
print("\n" + "=" * 60)
print("Section 2: Bin 9 vs non-bin9 KS test")
print("=" * 60)

bin9_df = train_df[train_df['is_bin9'] == 1]
non_bin9_df = train_df[train_df['is_bin9'] == 0]

numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
exclude = ['avg_delay_minutes_next_30m', 'oof', 'residual', 'is_bin9']
numeric_cols = [c for c in numeric_cols if c not in exclude]

ks_results = []
for col in numeric_cols:
    a = bin9_df[col].dropna().values
    b = non_bin9_df[col].dropna().values
    if len(a) > 50 and len(b) > 50:
        ks, p = ks_2samp(a, b)
        ks_results.append({
            'feature': col,
            'ks': ks,
            'p_value': p,
            'bin9_mean': a.mean(),
            'non_bin9_mean': b.mean(),
            'bin9_median': np.median(a),
            'non_bin9_median': np.median(b),
            'bin9_std': a.std(),
            'non_bin9_std': b.std(),
            'ratio': a.mean() / (b.mean() + 1e-9),
        })

ks_df = pd.DataFrame(ks_results).sort_values('ks', ascending=False)
ks_df.to_csv('output/phase13s4_bin9/bin9_ks_ranking.csv', index=False)

print("\n=== Top 30 features distinguishing bin 9 from others ===")
print(ks_df.head(30).to_string(index=False))

strong_separators = ks_df[ks_df['ks'] > 0.5]
print(f"\nFeatures with KS > 0.5: {len(strong_separators)}")
if len(strong_separators) > 0:
    print("-> Strong separation possible. Bin 9 may be predictable.")
    for _, row in strong_separators.iterrows():
        print(f"  {row['feature']}: KS={row['ks']:.3f}, bin9_mean={row['bin9_mean']:.3f}, non_bin9_mean={row['non_bin9_mean']:.3f}")
else:
    print("-> No strong separators. Bin 9 might be close to noise.")

medium_separators = ks_df[(ks_df['ks'] > 0.3) & (ks_df['ks'] <= 0.5)]
print(f"Features with 0.3 < KS <= 0.5: {len(medium_separators)}")

# ============================================================
# Section 3: Bin 9 clustering
# ============================================================
print("\n" + "=" * 60)
print("Section 3: Bin 9 clustering")
print("=" * 60)

top_features = ks_df.head(10)['feature'].tolist()
print(f"Top 10 features for clustering: {top_features}")

X_bin9 = bin9_df[top_features].fillna(0).values
scaler_clust = StandardScaler()
X_bin9_scaled = scaler_clust.fit_transform(X_bin9)

for n_clusters in [2, 3, 4, 5]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_bin9_scaled)

    print(f"\n=== KMeans n_clusters={n_clusters} ===")
    for c in range(n_clusters):
        mask = labels == c
        n = mask.sum()
        target_mean = bin9_df.iloc[mask]['avg_delay_minutes_next_30m'].mean()
        target_std = bin9_df.iloc[mask]['avg_delay_minutes_next_30m'].std()
        residual_mean = bin9_df.iloc[mask]['residual'].mean()
        print(f"  Cluster {c}: n={n}, target_mean={target_mean:.1f}, target_std={target_std:.1f}, residual={residual_mean:.1f}")
        # Print cluster centers for top 3 features
        if n_clusters == 3:
            center = km.cluster_centers_[c]
            for fi, feat in enumerate(top_features[:5]):
                print(f"    {feat}: center={center[fi]:.2f}")

# ============================================================
# Section 4: Bin 9 occurrence conditions
# ============================================================
print("\n" + "=" * 60)
print("Section 4: Bin 9 occurrence conditions")
print("=" * 60)

top5 = ks_df.head(5)['feature'].tolist()
print(f"=== Bin 9 occurrence by top 5 feature quantiles ===")

for feat in top5:
    train_df['_q'] = pd.qcut(train_df[feat], q=10, labels=False, duplicates='drop')
    bin9_rate = train_df.groupby('_q')['is_bin9'].agg(['mean', 'sum', 'count'])
    bin9_rate.columns = ['bin9_rate', 'bin9_count', 'total']
    print(f"\n{feat}:")
    print(bin9_rate.to_string())

    max_q = bin9_rate['bin9_rate'].idxmax()
    print(f"  -> q{max_q}: bin 9 rate = {bin9_rate.loc[max_q, 'bin9_rate']*100:.1f}%")

train_df = train_df.drop(columns=['_q'])

# 2-feature combination
print("\n=== Bin 9 rate by 2-feature combination ===")
top2 = ks_df.head(2)['feature'].tolist()
train_df['q1'] = pd.qcut(train_df[top2[0]], q=5, labels=False, duplicates='drop')
train_df['q2'] = pd.qcut(train_df[top2[1]], q=5, labels=False, duplicates='drop')

heatmap_data = train_df.groupby(['q1', 'q2'])['is_bin9'].mean().unstack()
print(f"Bin 9 rate by ({top2[0]} q1) x ({top2[1]} q2):")
print(heatmap_data.round(3).to_string())

# Find the max bin 9 rate cell
max_rate = heatmap_data.max().max()
print(f"\nMax bin 9 rate in 2-feature grid: {max_rate*100:.1f}%")

train_df = train_df.drop(columns=['q1', 'q2'])

# ============================================================
# Section 5: Layout / Scenario / Time distribution
# ============================================================
print("\n" + "=" * 60)
print("Section 5: Layout / Scenario / Time distribution")
print("=" * 60)

# Layout type
print("\n=== Bin 9 distribution by layout_type ===")
print(train_df.groupby('layout_type')['is_bin9'].agg(['mean', 'sum', 'count']).to_string())

# Layout ID concentration
print("\n=== Top 20 layouts with highest bin 9 rate ===")
layout_bin9 = train_df.groupby('layout_id').agg(
    bin9_rate=('is_bin9', 'mean'),
    bin9_count=('is_bin9', 'sum'),
    total=('is_bin9', 'count'),
    target_mean=('avg_delay_minutes_next_30m', 'mean'),
)
layout_bin9 = layout_bin9.sort_values('bin9_rate', ascending=False)
print(layout_bin9.head(20).to_string())
print(f"\nLayouts with bin 9 rate > 20%: {(layout_bin9['bin9_rate'] > 0.2).sum()}")
print(f"Layouts with bin 9 rate = 0%: {(layout_bin9['bin9_rate'] == 0).sum()}")
print(f"Layouts with bin 9 rate > 0 and <= 5%: {((layout_bin9['bin9_rate'] > 0) & (layout_bin9['bin9_rate'] <= 0.05)).sum()}")
layout_bin9.to_csv('output/phase13s4_bin9/bin9_by_layout.csv')

# Scenario distribution
print("\n=== Bin 9 by scenario ===")
scenario_bin9 = train_df.groupby('scenario_id')['is_bin9'].sum()
print(f"Scenarios with 0 bin 9: {(scenario_bin9 == 0).sum()}")
print(f"Scenarios with 1 bin 9: {(scenario_bin9 == 1).sum()}")
print(f"Scenarios with 2+ bin 9: {(scenario_bin9 >= 2).sum()}")
print(f"Max bin 9 in single scenario: {scenario_bin9.max()}")
print(f"Mean: {scenario_bin9.mean():.2f}")

# Time progression within scenario
print("\n=== Bin 9 in scenario time progression ===")
train_sorted = train_df.sort_values(['scenario_id']).copy()
train_sorted['ts'] = train_sorted.groupby('scenario_id').cumcount()
ts_bin9 = train_sorted.groupby('ts')['is_bin9'].mean()
print("Time step -> bin 9 rate:")
print(ts_bin9.to_string())

# ============================================================
# Section 6: Bin 9 classification (predictability)
# ============================================================
print("\n" + "=" * 60)
print("Section 6: Can we predict bin 9 occurrence?")
print("=" * 60)

top30 = ks_df.head(30)['feature'].tolist()
X_cls = train_df[top30].fillna(0).values
y_bin9 = train_df['is_bin9'].values

# LogisticRegression with GroupKFold
gkf = GroupKFold(n_splits=5)
oof_proba = np.zeros(len(X_cls))

for fold, (tr, va) in enumerate(gkf.split(X_cls, y_bin9, groups=train_df['layout_id'])):
    m = LogisticRegression(max_iter=1000, class_weight='balanced')
    m.fit(X_cls[tr], y_bin9[tr])
    oof_proba[va] = m.predict_proba(X_cls[va])[:, 1]

auc_lr = roc_auc_score(y_bin9, oof_proba)
print(f"Bin 9 classification AUC (LogReg, top 30 KS features): {auc_lr:.4f}")

# LightGBM
oof_proba_lgb = np.zeros(len(X_cls))

for fold, (tr, va) in enumerate(gkf.split(X_cls, y_bin9, groups=train_df['layout_id'])):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                            verbosity=-1, class_weight='balanced')
    m.fit(X_cls[tr], y_bin9[tr])
    oof_proba_lgb[va] = m.predict_proba(X_cls[va])[:, 1]

auc_lgb = roc_auc_score(y_bin9, oof_proba_lgb)
print(f"Bin 9 classification AUC (LGBM, top 30 KS features): {auc_lgb:.4f}")

print(f"\n-> AUC > 0.85: bin 9 predictable, new features / 2-stage model possible")
print(f"-> AUC 0.7-0.85: partially predictable, may help")
print(f"-> AUC < 0.7: bin 9 is nearly random, fundamental limit")

# Precision by decile
proba_df = pd.DataFrame({'proba': oof_proba_lgb, 'y': y_bin9})
proba_df['proba_bin'] = pd.qcut(proba_df['proba'], q=10, labels=False, duplicates='drop')
precision_table = proba_df.groupby('proba_bin').agg(
    n=('y', 'count'),
    bin9_actual=('y', 'sum'),
    precision=('y', 'mean'),
    avg_proba=('proba', 'mean')
)
print(f"\n=== LGBM bin 9 prediction precision by decile ===")
print(precision_table.to_string())

# ============================================================
# Section 7: Test set bin 9 probability
# ============================================================
print("\n" + "=" * 60)
print("Section 7: Test set bin 9 probability")
print("=" * 60)

test_df = pd.read_csv('data/test.csv').merge(layout_info, on='layout_id', how='left')

# Train full classifier
m_final = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              verbosity=-1, class_weight='balanced', random_state=42)
m_final.fit(X_cls, y_bin9)

X_test_cls = test_df[top30].fillna(0).values
test_proba = m_final.predict_proba(X_test_cls)[:, 1]

print(f"Test bin 9 probability distribution:")
print(f"  mean: {test_proba.mean():.4f}")
print(f"  std: {test_proba.std():.4f}")
print(f"  >0.5: {(test_proba > 0.5).sum()} ({(test_proba > 0.5).mean()*100:.2f}%)")
print(f"  >0.3: {(test_proba > 0.3).sum()} ({(test_proba > 0.3).mean()*100:.2f}%)")
print(f"  >0.1: {(test_proba > 0.1).sum()} ({(test_proba > 0.1).mean()*100:.2f}%)")

train_proba = m_final.predict_proba(X_cls)[:, 1]
print(f"\nTrain bin 9 probability distribution (for comparison):")
print(f"  mean: {train_proba.mean():.4f}")
print(f"  >0.5: {(train_proba > 0.5).mean()*100:.2f}%")
print(f"  >0.3: {(train_proba > 0.3).mean()*100:.2f}%")
print(f"  >0.1: {(train_proba > 0.1).mean()*100:.2f}%")

print(f"\nIf test has more high-proba samples -> test MAE would be higher (Public bias)")

# ============================================================
# Section 8: Visualization
# ============================================================
print("\n" + "=" * 60)
print("Section 8: Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1) bin 9 vs non-bin9 target distribution
axes[0, 0].hist(non_bin9_df['avg_delay_minutes_next_30m'], bins=50, alpha=0.5,
                label='non-bin9', color='blue', density=True)
axes[0, 0].hist(bin9_df['avg_delay_minutes_next_30m'], bins=50, alpha=0.5,
                label='bin9', color='red', density=True)
axes[0, 0].set_xlabel('Target')
axes[0, 0].set_title(f'Target distribution (threshold={THRESHOLD})')
axes[0, 0].legend()

# 2-6) Top 5 KS feature distributions
top_ks = ks_df.head(5)['feature'].tolist()
plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
for i, feat in enumerate(top_ks):
    r, c = plot_positions[i]
    ax = axes[r, c]
    a = bin9_df[feat].dropna()
    b = non_bin9_df[feat].dropna()
    ax.hist(b, bins=40, alpha=0.5, label='non-bin9', color='blue', density=True)
    ax.hist(a, bins=40, alpha=0.5, label='bin9', color='red', density=True)
    ax.set_xlabel(feat)
    ks_val = ks_df[ks_df['feature'] == feat]['ks'].values[0]
    ax.set_title(f'{feat} (KS={ks_val:.3f})')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('output/phase13s4_bin9/bin9_analysis.png', dpi=100)
plt.close()
print("  bin9_analysis.png saved")

# ROC curve
fig, ax = plt.subplots(figsize=(8, 8))
fpr_lr, tpr_lr, _ = roc_curve(y_bin9, oof_proba)
fpr_lgb, tpr_lgb, _ = roc_curve(y_bin9, oof_proba_lgb)
ax.plot(fpr_lr, tpr_lr, label=f'LogReg AUC={auc_lr:.4f}')
ax.plot(fpr_lgb, tpr_lgb, label=f'LGBM AUC={auc_lgb:.4f}')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Bin 9 Classification ROC')
ax.legend()
plt.savefig('output/phase13s4_bin9/bin9_roc.png', dpi=100)
plt.close()
print("  bin9_roc.png saved")

# Time step bin 9 rate
fig, axes2 = plt.subplots(1, 2, figsize=(14, 5))
axes2[0].bar(ts_bin9.index, ts_bin9.values, color='coral')
axes2[0].set_xlabel('Time step in scenario')
axes2[0].set_ylabel('Bin 9 rate')
axes2[0].set_title('Bin 9 rate by time step')

# Test vs train bin 9 probability histogram
axes2[1].hist(train_proba, bins=50, alpha=0.5, label='Train', density=True, color='blue')
axes2[1].hist(test_proba, bins=50, alpha=0.5, label='Test', density=True, color='orange')
axes2[1].set_xlabel('Bin 9 probability')
axes2[1].set_ylabel('Density')
axes2[1].set_title('Bin 9 probability: Train vs Test')
axes2[1].legend()

plt.tight_layout()
plt.savefig('output/phase13s4_bin9/bin9_time_and_test.png', dpi=100)
plt.close()
print("  bin9_time_and_test.png saved")

# ============================================================
# Section 9: Summary
# ============================================================
print("\n" + "=" * 60)
print("Section 9: Generating summary")
print("=" * 60)

summary = []
summary.append("# Phase 13 Step 4: Bin 9 EDA Summary")
summary.append("")
summary.append("## 1. Bin 9 Definition and Scale")
summary.append(f"- Threshold: target > {THRESHOLD}")
summary.append(f"- Train ratio: {n_bin9/len(train_df)*100:.2f}% ({n_bin9} samples)")
summary.append(f"- Model avg prediction: {bin9_oof_mean:.1f} (actual avg: {y[y>THRESHOLD].mean():.1f})")
summary.append(f"- Residual: +{bin9_residual_mean:.1f} (Phase 13s1 OOF)")
summary.append("")

summary.append("## 2. Bin 9 vs non-bin9 Separability")
summary.append("")
summary.append("### Top 10 KS features")
summary.append("| Feature | KS | bin9_mean | non_bin9_mean | ratio |")
summary.append("|---------|-----|-----------|---------------|-------|")
for _, row in ks_df.head(10).iterrows():
    summary.append(f"| {row['feature']} | {row['ks']:.3f} | {row['bin9_mean']:.3f} | {row['non_bin9_mean']:.3f} | {row['ratio']:.2f} |")
summary.append("")
summary.append(f"### Strong separators (KS > 0.5): {len(strong_separators)}")
summary.append(f"### Medium separators (0.3 < KS <= 0.5): {len(medium_separators)}")
summary.append("")

summary.append("## 3. Bin 9 Predictability (Classifier)")
summary.append(f"- LogReg AUC: {auc_lr:.4f}")
summary.append(f"- LGBM AUC: {auc_lgb:.4f}")
summary.append("- Conclusion:")
if auc_lgb > 0.85:
    summary.append("  - **AUC > 0.85**: Bin 9 is predictable! 2-stage model recommended.")
elif auc_lgb > 0.7:
    summary.append("  - **AUC 0.7-0.85**: Partially predictable. Careful feature engineering may help.")
else:
    summary.append("  - **AUC < 0.7**: Bin 9 is nearly random. Fundamental limit.")
summary.append("")

summary.append("## 4. Bin 9 Occurrence Patterns")
summary.append(f"- Layouts with bin 9 rate > 20%: {(layout_bin9['bin9_rate'] > 0.2).sum()}")
summary.append(f"- Layouts with bin 9 rate = 0%: {(layout_bin9['bin9_rate'] == 0).sum()}")
sc_0 = (scenario_bin9 == 0).sum()
sc_1 = (scenario_bin9 == 1).sum()
sc_2p = (scenario_bin9 >= 2).sum()
summary.append(f"- Scenarios with 0 bin 9: {sc_0}, 1: {sc_1}, 2+: {sc_2p}")
summary.append(f"- Mean bin 9 per scenario: {scenario_bin9.mean():.2f}")
summary.append(f"- Max bin 9 in single scenario: {scenario_bin9.max()}")
summary.append("")

summary.append("## 5. Test Set Impact")
summary.append(f"- Test samples with bin 9 prob > 0.5: {(test_proba > 0.5).mean()*100:.2f}% (train: {(train_proba > 0.5).mean()*100:.2f}%)")
summary.append(f"- Test samples with bin 9 prob > 0.3: {(test_proba > 0.3).mean()*100:.2f}% (train: {(train_proba > 0.3).mean()*100:.2f}%)")
summary.append(f"- Test samples with bin 9 prob > 0.1: {(test_proba > 0.1).mean()*100:.2f}% (train: {(train_proba > 0.1).mean()*100:.2f}%)")
summary.append("")

summary.append("## 6. Phase 13 Step 5 Strategy")
summary.append("")
summary.append("### Scenario A: AUC > 0.85 (predictable)")
summary.append("- 2-Stage prediction: Stage 1 bin 9 classifier, Stage 2 specialized regressor")
summary.append("- Expected: bin 9 residual +95 -> +30~50")
summary.append("")
summary.append("### Scenario B: AUC 0.7~0.85 (partially predictable)")
summary.append("- Soft weighting + nonlinear interaction features from top KS features")
summary.append("")
summary.append("### Scenario C: AUC < 0.7 (fundamental limit)")
summary.append("- Accept bin 9 as noise, focus on normal range precision")
summary.append("- Target adjustment: 5th -> 10th place")
summary.append("")
summary.append("## Conclusion")
summary.append("(To be filled after reviewing AUC and pattern results)")
summary.append("")
summary.append("---")
summary.append("*Generated by run_phase13s4_bin9_eda.py*")

with open('output/phase13s4_bin9/summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary))
print("  summary.md saved")

print("\n" + "=" * 60)
print("All analysis saved to output/phase13s4_bin9/")
print("  - bin9_ks_ranking.csv")
print("  - bin9_by_layout.csv")
print("  - bin9_analysis.png")
print("  - bin9_roc.png")
print("  - bin9_time_and_test.png")
print("  - summary.md")
print("=" * 60)
