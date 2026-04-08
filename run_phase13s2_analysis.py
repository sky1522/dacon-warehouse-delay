"""
Phase 13 Step 2: Hard Layout Analysis
- Layout-level MAE decomposition from Phase 13s1 OOF
- Hard vs Easy layout characterization
- Test set similarity analysis
- Strategy recommendations for Phase 13 Step 3
"""

import pickle
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

os.makedirs('output/phase13s2_analysis', exist_ok=True)

# ============================================================
# Section 1: Load OOF and compute residuals
# ============================================================
print("=" * 60)
print("Section 1: Load OOF and compute residuals")
print("=" * 60)

with open('output/ckpt_phase13s1_lgb_huber.pkl', 'rb') as f:
    ckpt = pickle.load(f)

oof = ckpt['oof']  # shape (250000,)
train_df = pd.read_csv('data/train.csv')
layout_info = pd.read_csv('data/layout_info.csv')

train_df = train_df.merge(layout_info, on='layout_id', how='left')
train_df['oof'] = oof
train_df['residual'] = train_df['avg_delay_minutes_next_30m'] - train_df['oof']
train_df['abs_residual'] = np.abs(train_df['residual'])

print(f"Train shape: {train_df.shape}")
print(f"OOF shape: {len(oof)}")
print(f"Overall MAE: {train_df['abs_residual'].mean():.4f}")
print(f"Overall residual mean: {train_df['residual'].mean():.4f}")

# ============================================================
# Section 2: Layout-level MAE ranking
# ============================================================
print("\n" + "=" * 60)
print("Section 2: Layout-level MAE ranking")
print("=" * 60)

layout_stats = train_df.groupby('layout_id').agg(
    mae=('abs_residual', 'mean'),
    residual_mean=('residual', 'mean'),
    residual_std=('residual', 'std'),
    target_mean=('avg_delay_minutes_next_30m', 'mean'),
    target_std=('avg_delay_minutes_next_30m', 'std'),
    target_max=('avg_delay_minutes_next_30m', 'max'),
    target_q99=('avg_delay_minutes_next_30m', lambda x: x.quantile(0.99)),
    n_samples=('avg_delay_minutes_next_30m', 'size'),
    layout_type=('layout_type', 'first'),
    robot_total=('robot_total', 'first'),
    pack_station_count=('pack_station_count', 'first'),
    charger_count=('charger_count', 'first'),
    floor_area_sqm=('floor_area_sqm', 'first'),
).reset_index()

layout_stats = layout_stats.sort_values('mae', ascending=False)
layout_stats.to_csv('output/phase13s2_analysis/layout_mae_ranking.csv', index=False)

# Hard/Easy thresholds
hard_threshold = layout_stats['mae'].quantile(0.8)
easy_threshold = layout_stats['mae'].quantile(0.2)
layout_stats['difficulty'] = 'medium'
layout_stats.loc[layout_stats['mae'] > hard_threshold, 'difficulty'] = 'hard'
layout_stats.loc[layout_stats['mae'] < easy_threshold, 'difficulty'] = 'easy'

n_hard = (layout_stats['difficulty'] == 'hard').sum()
n_easy = (layout_stats['difficulty'] == 'easy').sum()
n_medium = (layout_stats['difficulty'] == 'medium').sum()

print(f"Hard layouts threshold: MAE > {hard_threshold:.3f}")
print(f"Easy layouts threshold: MAE < {easy_threshold:.3f}")
print(f"Hard: {n_hard}, Medium: {n_medium}, Easy: {n_easy}")
print(f"\nTop 10 hardest layouts:")
print(layout_stats.head(10)[['layout_id', 'mae', 'residual_mean', 'target_mean', 'target_std', 'n_samples', 'layout_type']].to_string(index=False))
print(f"\nTop 10 easiest layouts:")
print(layout_stats.tail(10)[['layout_id', 'mae', 'residual_mean', 'target_mean', 'target_std', 'n_samples', 'layout_type']].to_string(index=False))

# ============================================================
# Section 3: Hard vs Easy layout comparison
# ============================================================
print("\n" + "=" * 60)
print("Section 3: Hard vs Easy layout comparison")
print("=" * 60)

hard_ids = layout_stats[layout_stats['difficulty'] == 'hard']['layout_id'].tolist()
easy_ids = layout_stats[layout_stats['difficulty'] == 'easy']['layout_id'].tolist()

hard_samples = train_df[train_df['layout_id'].isin(hard_ids)]
easy_samples = train_df[train_df['layout_id'].isin(easy_ids)]

print(f"Hard samples: {len(hard_samples)}")
print(f"Easy samples: {len(easy_samples)}")

# 1) Layout_type distribution
print("\n=== Layout type distribution ===")
print("Hard:")
print(hard_samples['layout_type'].value_counts(normalize=True).to_string())
print("\nEasy:")
print(easy_samples['layout_type'].value_counts(normalize=True).to_string())

# 2) Target distribution
TARGET = 'avg_delay_minutes_next_30m'
print("\n=== Target distribution ===")
print(f"Hard: mean={hard_samples[TARGET].mean():.2f}, "
      f"std={hard_samples[TARGET].std():.2f}, "
      f"q99={hard_samples[TARGET].quantile(0.99):.2f}, "
      f"max={hard_samples[TARGET].max():.2f}")
print(f"Easy: mean={easy_samples[TARGET].mean():.2f}, "
      f"std={easy_samples[TARGET].std():.2f}, "
      f"q99={easy_samples[TARGET].quantile(0.99):.2f}, "
      f"max={easy_samples[TARGET].max():.2f}")

# 3) KS test all numeric columns (hard vs easy)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
exclude = [TARGET, 'oof', 'residual', 'abs_residual']
numeric_cols = [c for c in numeric_cols if c not in exclude]

ks_results = []
for col in numeric_cols:
    h = hard_samples[col].dropna().values
    e = easy_samples[col].dropna().values
    if len(h) > 100 and len(e) > 100:
        ks, p = ks_2samp(h, e)
        ks_results.append({
            'feature': col,
            'ks': ks,
            'p_value': p,
            'hard_mean': h.mean(),
            'easy_mean': e.mean(),
            'hard_std': h.std(),
            'easy_std': e.std(),
        })

ks_df = pd.DataFrame(ks_results).sort_values('ks', ascending=False)
ks_df.to_csv('output/phase13s2_analysis/hard_vs_easy_ks.csv', index=False)

print("\n=== Top 30 features distinguishing Hard vs Easy ===")
print(ks_df.head(30).to_string(index=False))

# ============================================================
# Section 4: Residual pattern in hard layouts
# ============================================================
print("\n" + "=" * 60)
print("Section 4: Residual pattern in hard layouts")
print("=" * 60)

# Residual bias by difficulty
print("\n=== Residual bias by difficulty ===")
for diff in ['hard', 'medium', 'easy']:
    subset = train_df[train_df['layout_id'].isin(
        layout_stats[layout_stats['difficulty'] == diff]['layout_id'])]
    print(f"{diff:8s}: residual mean = {subset['residual'].mean():+.3f} "
          f"(positive = model underpredicts), "
          f"abs_residual mean = {subset['abs_residual'].mean():.3f}")

# Target bin residual (hard layouts only)
hard_samples = hard_samples.copy()
hard_samples['target_bin'] = pd.qcut(
    hard_samples[TARGET],
    q=10, labels=False, duplicates='drop'
)
bin_residual = hard_samples.groupby('target_bin').agg(
    target_mean=(TARGET, 'mean'),
    oof_mean=('oof', 'mean'),
    residual_mean=('residual', 'mean'),
    abs_residual_mean=('abs_residual', 'mean'),
    n=('residual', 'size')
)
print("\n=== Hard layouts: Residual by target bin ===")
print(bin_residual.to_string())

# Extreme target concentration
train_q95 = train_df[TARGET].quantile(0.95)
extreme_in_hard = (hard_samples[TARGET] > train_q95).mean()
extreme_in_easy = (easy_samples[TARGET] > train_q95).mean()
extreme_all = (train_df[TARGET] > train_q95).mean()
print(f"\nExtreme target (>q95={train_q95:.1f}) ratio:")
print(f"  Hard layouts: {extreme_in_hard*100:.2f}%")
print(f"  Easy layouts: {extreme_in_easy*100:.2f}%")
print(f"  Global: {extreme_all*100:.2f}%")

# ============================================================
# Section 5: Layout_info static features - Hard vs Easy
# ============================================================
print("\n" + "=" * 60)
print("Section 5: Layout_info static features - Hard vs Easy")
print("=" * 60)

hard_layout_stats = layout_stats[layout_stats['difficulty'] == 'hard']
easy_layout_stats = layout_stats[layout_stats['difficulty'] == 'easy']

static_cols = ['robot_total', 'pack_station_count', 'charger_count', 'floor_area_sqm']
print("\n=== Layout static features: Hard vs Easy ===")
for col in static_cols:
    h = hard_layout_stats[col].mean()
    e = easy_layout_stats[col].mean()
    ratio = h / e if e != 0 else float('inf')
    print(f"  {col}: hard={h:.2f}, easy={e:.2f}, ratio={ratio:.2f}")

# Layout_type by MAE
print("\n=== Mean MAE by layout_type ===")
type_mae = layout_stats.groupby('layout_type')['mae'].agg(['mean', 'std', 'count'])
print(type_mae.to_string())

# Full layout_info comparison for hard vs easy
all_layout_cols = [c for c in layout_info.columns if c not in ['layout_id', 'layout_type']]
print("\n=== All layout_info features: Hard vs Easy ===")
for col in all_layout_cols:
    if col in hard_layout_stats.columns:
        h = hard_layout_stats[col].mean()
        e = easy_layout_stats[col].mean()
    else:
        h_layouts = layout_info[layout_info['layout_id'].isin(hard_ids)]
        e_layouts = layout_info[layout_info['layout_id'].isin(easy_ids)]
        h = h_layouts[col].mean() if col in h_layouts.columns else np.nan
        e = e_layouts[col].mean() if col in e_layouts.columns else np.nan
    if not np.isnan(h) and not np.isnan(e) and e != 0:
        print(f"  {col:25s}: hard={h:.3f}, easy={e:.3f}, ratio={h/e:.3f}")

# ============================================================
# Section 6: Test set similarity - MOST IMPORTANT
# ============================================================
print("\n" + "=" * 60)
print("Section 6: Test set similarity - MOST IMPORTANT")
print("=" * 60)

test_df = pd.read_csv('data/test.csv').merge(layout_info, on='layout_id', how='left')

# Test layout static distribution
test_layout_stats = test_df.groupby('layout_id').agg(
    robot_total=('robot_total', 'first'),
    pack_station_count=('pack_station_count', 'first'),
    charger_count=('charger_count', 'first'),
    floor_area_sqm=('floor_area_sqm', 'first'),
    layout_type=('layout_type', 'first'),
).reset_index()

print("\n=== Test layouts: static distribution ===")
closer_to_hard = 0
closer_to_easy = 0
for col in static_cols:
    t_mean = test_layout_stats[col].mean()
    h_mean = hard_layout_stats[col].mean()
    e_mean = easy_layout_stats[col].mean()
    a_mean = layout_stats[col].mean()
    closer_to = 'HARD' if abs(t_mean - h_mean) < abs(t_mean - e_mean) else 'EASY'
    if closer_to == 'HARD':
        closer_to_hard += 1
    else:
        closer_to_easy += 1
    print(f"  {col}: test={t_mean:.2f}, hard_train={h_mean:.2f}, easy_train={e_mean:.2f}, all_train={a_mean:.2f}")
    print(f"    -> Test is closer to: {closer_to}")

print(f"\nOverall: Test closer to HARD in {closer_to_hard}/{len(static_cols)} features")

print("\n=== Test layout_type distribution ===")
print(f"  Test:       {test_layout_stats['layout_type'].value_counts(normalize=True).to_dict()}")
print(f"  Hard train: {hard_layout_stats['layout_type'].value_counts(normalize=True).to_dict()}")
print(f"  Easy train: {easy_layout_stats['layout_type'].value_counts(normalize=True).to_dict()}")
print(f"  All train:  {layout_stats['layout_type'].value_counts(normalize=True).to_dict()}")

# Test layouts overlap with train
train_layouts = set(layout_stats['layout_id'])
test_layouts = set(test_layout_stats['layout_id'])
overlap = train_layouts & test_layouts
test_only = test_layouts - train_layouts
print(f"\n=== Layout overlap ===")
print(f"  Train layouts: {len(train_layouts)}")
print(f"  Test layouts: {len(test_layouts)}")
print(f"  Overlap: {len(overlap)}")
print(f"  Test-only (unseen): {len(test_only)}")

# For overlapping layouts, what's their difficulty?
if len(overlap) > 0:
    overlap_diff = layout_stats[layout_stats['layout_id'].isin(overlap)]['difficulty'].value_counts()
    print(f"  Overlapping layouts difficulty: {overlap_diff.to_dict()}")

# Compare dynamic feature distributions: test vs hard vs easy
print("\n=== Dynamic features: Test vs Hard vs Easy (sample features) ===")
key_dynamic = ['order_inflow_15m', 'robot_active', 'congestion_score',
               'battery_mean', 'pack_utilization', 'loading_dock_util',
               'charge_queue_length', 'low_battery_ratio']
for col in key_dynamic:
    if col in test_df.columns and col in hard_samples.columns:
        t = test_df[col].dropna()
        h = hard_samples[col].dropna()
        e = easy_samples[col].dropna()
        ks_th, _ = ks_2samp(t, h)
        ks_te, _ = ks_2samp(t, e)
        closer = 'HARD' if ks_th < ks_te else 'EASY'
        print(f"  {col:25s}: KS(test,hard)={ks_th:.3f}, KS(test,easy)={ks_te:.3f} -> Test closer to {closer}")

# ============================================================
# Section 7: Visualization
# ============================================================
print("\n" + "=" * 60)
print("Section 7: Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1) Layout MAE distribution
axes[0, 0].hist(layout_stats['mae'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(hard_threshold, color='red', linestyle='--', label=f'Hard threshold ({hard_threshold:.2f})')
axes[0, 0].axvline(easy_threshold, color='green', linestyle='--', label=f'Easy threshold ({easy_threshold:.2f})')
axes[0, 0].set_xlabel('Layout MAE')
axes[0, 0].set_title('Distribution of per-layout MAE (Phase 13s1 OOF)')
axes[0, 0].legend(fontsize=8)

# 2) Target distribution: hard vs easy
axes[0, 1].hist(hard_samples[TARGET], bins=50,
                alpha=0.5, label='Hard', density=True, color='red')
axes[0, 1].hist(easy_samples[TARGET], bins=50,
                alpha=0.5, label='Easy', density=True, color='green')
axes[0, 1].set_xlabel('Target')
axes[0, 1].set_title('Target distribution: Hard vs Easy layouts')
axes[0, 1].set_xlim(0, 150)
axes[0, 1].legend()

# 3) Residual by target bin (hard)
axes[0, 2].bar(bin_residual.index, bin_residual['residual_mean'], color='salmon')
axes[0, 2].set_xlabel('Target bin (0=low, 9=high)')
axes[0, 2].set_ylabel('Residual mean')
axes[0, 2].set_title('Hard layouts: prediction bias by target level')
axes[0, 2].axhline(0, color='black', linewidth=0.5)

# 4-6) Top 3 KS features: hard vs easy distributions
top_ks_features = ks_df.head(5)['feature'].tolist()
for i, feat in enumerate(top_ks_features[:3]):
    ax = axes[1, i]
    h = hard_samples[feat].dropna()
    e = easy_samples[feat].dropna()
    ax.hist(h, bins=40, alpha=0.5, label='Hard', density=True, color='red')
    ax.hist(e, bins=40, alpha=0.5, label='Easy', density=True, color='green')
    ax.set_xlabel(feat)
    ks_val = ks_df[ks_df['feature'] == feat]['ks'].values[0]
    ax.set_title(f'{feat}: KS={ks_val:.3f}')
    ax.legend()

plt.tight_layout()
plt.savefig('output/phase13s2_analysis/hard_layout_analysis.png', dpi=100)
plt.close()
print("  hard_layout_analysis.png saved")

# Additional: scatter MAE vs target_mean per layout
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

colors_diff = {'hard': 'red', 'medium': 'gray', 'easy': 'green'}
for diff in ['easy', 'medium', 'hard']:
    subset = layout_stats[layout_stats['difficulty'] == diff]
    axes2[0].scatter(subset['target_mean'], subset['mae'], alpha=0.5, s=15,
                     color=colors_diff[diff], label=diff)
axes2[0].set_xlabel('Target mean')
axes2[0].set_ylabel('MAE')
axes2[0].set_title('Layout: Target mean vs MAE')
axes2[0].legend()

for diff in ['easy', 'medium', 'hard']:
    subset = layout_stats[layout_stats['difficulty'] == diff]
    axes2[1].scatter(subset['target_std'], subset['mae'], alpha=0.5, s=15,
                     color=colors_diff[diff], label=diff)
axes2[1].set_xlabel('Target std')
axes2[1].set_ylabel('MAE')
axes2[1].set_title('Layout: Target std vs MAE')
axes2[1].legend()

for diff in ['easy', 'medium', 'hard']:
    subset = layout_stats[layout_stats['difficulty'] == diff]
    axes2[2].scatter(subset['n_samples'], subset['mae'], alpha=0.5, s=15,
                     color=colors_diff[diff], label=diff)
axes2[2].set_xlabel('N samples')
axes2[2].set_ylabel('MAE')
axes2[2].set_title('Layout: Sample count vs MAE')
axes2[2].legend()

plt.tight_layout()
plt.savefig('output/phase13s2_analysis/hard_layout_scatter.png', dpi=100)
plt.close()
print("  hard_layout_scatter.png saved")

# ============================================================
# Section 8: Summary file
# ============================================================
print("\n" + "=" * 60)
print("Section 8: Generating summary")
print("=" * 60)

summary = []
summary.append("# Phase 13 Step 2: Hard Layout Analysis Summary")
summary.append("")
summary.append("## 1. Layout MAE Distribution")
summary.append(f"- Total layouts analyzed: {len(layout_stats)}")
summary.append(f"- Hard threshold (q80): MAE > {hard_threshold:.3f}")
summary.append(f"- Easy threshold (q20): MAE < {easy_threshold:.3f}")
summary.append(f"- Hard: {n_hard} layouts, Medium: {n_medium}, Easy: {n_easy}")
summary.append(f"- MAE range: {layout_stats['mae'].min():.3f} ~ {layout_stats['mae'].max():.3f}")
summary.append(f"- Overall MAE: {train_df['abs_residual'].mean():.4f}")
summary.append("")

summary.append("## 2. Hard Layout Common Characteristics")
summary.append("Top 10 features distinguishing Hard vs Easy (KS test):")
summary.append("")
for _, row in ks_df.head(10).iterrows():
    summary.append(f"- {row['feature']}: KS={row['ks']:.3f}, "
                   f"hard_mean={row['hard_mean']:.3f}, easy_mean={row['easy_mean']:.3f}")
summary.append("")

summary.append("## 3. Target Distribution Difference")
summary.append(f"- Hard: mean={hard_samples[TARGET].mean():.2f}, "
               f"std={hard_samples[TARGET].std():.2f}, "
               f"q99={hard_samples[TARGET].quantile(0.99):.2f}, "
               f"max={hard_samples[TARGET].max():.2f}")
summary.append(f"- Easy: mean={easy_samples[TARGET].mean():.2f}, "
               f"std={easy_samples[TARGET].std():.2f}, "
               f"q99={easy_samples[TARGET].quantile(0.99):.2f}, "
               f"max={easy_samples[TARGET].max():.2f}")
summary.append(f"- Extreme target(>q95={train_q95:.1f}) ratio: "
               f"hard={extreme_in_hard*100:.2f}%, easy={extreme_in_easy*100:.2f}%")
summary.append("")

summary.append("## 4. Model Bias Pattern")
for diff in ['hard', 'medium', 'easy']:
    subset = train_df[train_df['layout_id'].isin(
        layout_stats[layout_stats['difficulty'] == diff]['layout_id'])]
    summary.append(f"- {diff}: residual mean = {subset['residual'].mean():+.3f}, "
                   f"MAE = {subset['abs_residual'].mean():.3f}")
summary.append("")
summary.append("Hard layout target bin residuals:")
for idx, row in bin_residual.iterrows():
    summary.append(f"  bin {idx}: target_mean={row['target_mean']:.1f}, "
                   f"residual_mean={row['residual_mean']:+.2f}, n={int(row['n'])}")
summary.append("")

summary.append("## 5. Test Set Similarity")
summary.append("")
for col in static_cols:
    t_mean = test_layout_stats[col].mean()
    h_mean = hard_layout_stats[col].mean()
    e_mean = easy_layout_stats[col].mean()
    closer_to = 'HARD' if abs(t_mean - h_mean) < abs(t_mean - e_mean) else 'EASY'
    summary.append(f"- {col}: test={t_mean:.2f}, hard={h_mean:.2f}, easy={e_mean:.2f} -> {closer_to}")
summary.append(f"- Layout overlap: {len(overlap)} layouts shared between train/test")
summary.append(f"- Test-only layouts: {len(test_only)}")
summary.append("")

summary.append("## 6. Phase 13 Step 3 Strategy Recommendations")
summary.append("")
summary.append("(To be filled after reviewing numerical results)")
summary.append("")
summary.append("### Potential directions:")
summary.append("- Hard layout specialized features (layout-aware interactions)")
summary.append("- Target encoding with smoothing on layout_id")
summary.append("- Layout difficulty as a meta-feature")
summary.append("- Separate models or sample weights for hard layouts")
summary.append("- Focus on reducing bias in high-target bins")
summary.append("")
summary.append("---")
summary.append("*Generated by run_phase13s2_analysis.py*")

with open('output/phase13s2_analysis/summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary))
print("  summary.md saved")

# Save all text output to a file
print("\n" + "=" * 60)
print("All analysis saved to output/phase13s2_analysis/")
print("  - layout_mae_ranking.csv")
print("  - hard_vs_easy_ks.csv")
print("  - hard_layout_analysis.png")
print("  - hard_layout_scatter.png")
print("  - summary.md")
print("=" * 60)
