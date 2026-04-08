"""
Comprehensive EDA for Phase 13 Strategy
- Target distribution, Train vs Test comparison, Adversarial validation
- Residual analysis, Correlation, Dynamic patterns, Summary
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.feature_selection import mutual_info_regression

# ============================================================
# Setup
# ============================================================
os.makedirs('output/eda_deep', exist_ok=True)

print("=" * 60, flush=True)
print("Loading data...", flush=True)
print("=" * 60, flush=True)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
layout_df = pd.read_csv('data/layout_info.csv')

TARGET = 'avg_delay_minutes_next_30m'

# Merge layout info
train_df = train_df.merge(layout_df, on='layout_id', how='left')
test_df = test_df.merge(layout_df, on='layout_id', how='left')

# Identify numeric columns (exclude ID-like columns)
id_cols = ['ID', 'layout_id', 'scenario_id']
cat_cols = ['layout_type']
num_cols = [c for c in train_df.columns
            if c not in id_cols + cat_cols + [TARGET]
            and train_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

print(f"Train: {train_df.shape}, Test: {test_df.shape}", flush=True)
print(f"Numeric columns: {len(num_cols)}", flush=True)
print(f"Target column: {TARGET}", flush=True)

# ============================================================
# Section 1: Target Distribution Analysis
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 1: Target Distribution Analysis", flush=True)
print("=" * 60, flush=True)

target = train_df[TARGET].dropna()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Section 1: Target Distribution Analysis', fontsize=16)

# Raw histogram
axes[0, 0].hist(target, bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Target Histogram')
axes[0, 0].set_xlabel(TARGET)
axes[0, 0].set_ylabel('Count')

# KDE
target.plot.kde(ax=axes[0, 1])
axes[0, 1].set_title('Target KDE')
axes[0, 1].set_xlabel(TARGET)

# log1p transform
log_target = np.log1p(target)
axes[0, 2].hist(log_target, bins=100, edgecolor='black', alpha=0.7, color='orange')
axes[0, 2].set_title('log1p(Target) Histogram')

# sqrt transform
sqrt_target = np.sqrt(target.clip(lower=0))
axes[1, 0].hist(sqrt_target, bins=100, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_title('sqrt(Target) Histogram')

# log1p KDE
log_target.plot.kde(ax=axes[1, 1], color='orange')
axes[1, 1].set_title('log1p(Target) KDE')

# sqrt KDE
sqrt_target.plot.kde(ax=axes[1, 2], color='green')
axes[1, 2].set_title('sqrt(Target) KDE')

plt.tight_layout()
plt.savefig('output/eda_deep/eda_01_target_distribution.png', dpi=100)
plt.close()

# Statistics
quantiles = target.quantile([0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
skewness = target.skew()
kurtosis = target.kurtosis()

# Outliers (IQR)
q1, q3 = target.quantile(0.25), target.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outlier_pct = ((target < lower_bound) | (target > upper_bound)).mean() * 100

# Target == 0
zero_pct = (target == 0).mean() * 100

# Extreme values (> q99)
q99 = target.quantile(0.99)
extreme_mask = target > q99
extreme_df = train_df[extreme_mask]

stats_text = []
stats_text.append("=== Target Statistics ===")
stats_text.append(f"Count: {len(target)}")
stats_text.append(f"Mean: {target.mean():.4f}")
stats_text.append(f"Std: {target.std():.4f}")
stats_text.append(f"Min: {target.min():.4f}")
stats_text.append(f"Max: {target.max():.4f}")
stats_text.append(f"Skewness: {skewness:.4f}")
stats_text.append(f"Kurtosis: {kurtosis:.4f}")
stats_text.append("")
stats_text.append("=== Quantiles ===")
for q_name, q_val in quantiles.items():
    stats_text.append(f"  q{int(q_name*100):02d}: {q_val:.4f}")
stats_text.append("")
stats_text.append(f"=== Outliers (IQR) ===")
stats_text.append(f"IQR: {iqr:.4f}")
stats_text.append(f"Lower bound: {lower_bound:.4f}")
stats_text.append(f"Upper bound: {upper_bound:.4f}")
stats_text.append(f"Outlier %: {outlier_pct:.2f}%")
stats_text.append("")
stats_text.append(f"=== Zero Target ===")
stats_text.append(f"Target == 0 ratio: {zero_pct:.2f}%")
stats_text.append(f"Target == 0 count: {(target == 0).sum()}")
stats_text.append("")
stats_text.append(f"=== Extreme Values (> q99 = {q99:.4f}) ===")
stats_text.append(f"Count: {extreme_mask.sum()}")
if len(extreme_df) > 0:
    stats_text.append("Mean of numeric features for extreme samples:")
    for c in num_cols[:20]:
        if c in extreme_df.columns:
            stats_text.append(f"  {c}: extreme_mean={extreme_df[c].mean():.4f}, all_mean={train_df[c].mean():.4f}")

with open('output/eda_deep/eda_01_target_stats.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(stats_text))

print("Section 1 done.", flush=True)

# ============================================================
# Section 2: Train vs Test Distribution Comparison
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 2: Train vs Test Distribution Comparison", flush=True)
print("=" * 60, flush=True)

# KS test for all numeric columns
ks_results = []
for c in num_cols:
    tr_vals = train_df[c].dropna().values
    te_vals = test_df[c].dropna().values
    if len(tr_vals) > 0 and len(te_vals) > 0:
        stat, pval = stats.ks_2samp(tr_vals, te_vals)
        ks_results.append({'feature': c, 'ks_stat': stat, 'p_value': pval})

ks_df = pd.DataFrame(ks_results).sort_values('ks_stat', ascending=False)

# Save KS ranking
ks_text = ["=== KS Test Ranking (Train vs Test) ===", ""]
ks_text.append(f"{'Rank':<6}{'Feature':<40}{'KS Stat':<12}{'P-value':<15}")
ks_text.append("-" * 73)
for i, row in ks_df.head(40).iterrows():
    rank = ks_df.index.get_loc(i) + 1
    ks_text.append(f"{rank:<6}{row['feature']:<40}{row['ks_stat']:<12.6f}{row['p_value']:<15.2e}")
ks_text.append("")
ks_text.append(f"Columns with p < 0.05: {(ks_df['p_value'] < 0.05).sum()} / {len(ks_df)}")
ks_text.append(f"Columns with KS > 0.1: {(ks_df['ks_stat'] > 0.1).sum()}")

with open('output/eda_deep/eda_02_train_test_ks_ranking.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(ks_text))

# KDE overlay plots - all columns (multiple pages)
n_per_page = 20
n_pages = (len(num_cols) + n_per_page - 1) // n_per_page

for page in range(n_pages):
    start = page * n_per_page
    end = min(start + n_per_page, len(num_cols))
    cols_page = num_cols[start:end]
    n_cols_page = len(cols_page)
    n_rows_fig = (n_cols_page + 3) // 4

    fig, axes = plt.subplots(n_rows_fig, 4, figsize=(20, 4 * n_rows_fig))
    axes = axes.flatten() if n_rows_fig > 1 else (axes.flatten() if n_cols_page > 1 else [axes])
    fig.suptitle(f'Train vs Test KDE (Page {page+1}/{n_pages})', fontsize=14)

    for idx, c in enumerate(cols_page):
        ax = axes[idx]
        tr_vals = train_df[c].dropna()
        te_vals = test_df[c].dropna()
        if len(tr_vals) > 10 and len(te_vals) > 10:
            try:
                tr_vals.plot.kde(ax=ax, label='Train', alpha=0.7)
                te_vals.plot.kde(ax=ax, label='Test', alpha=0.7)
            except Exception:
                ax.hist(tr_vals, bins=50, alpha=0.5, label='Train', density=True)
                ax.hist(te_vals, bins=50, alpha=0.5, label='Test', density=True)
        ax.set_title(c, fontsize=9)
        ax.legend(fontsize=7)

    for idx in range(len(cols_page), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'output/eda_deep/eda_02_train_test_kde_all_p{page+1}.png', dpi=100)
    plt.close()

# Top 10 different columns - detailed plots
top10_cols = ks_df.head(10)['feature'].tolist()
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()
fig.suptitle('Top 10 Most Different Columns (Train vs Test)', fontsize=16)

for idx, c in enumerate(top10_cols):
    ax = axes[idx]
    tr_vals = train_df[c].dropna()
    te_vals = test_df[c].dropna()
    try:
        tr_vals.plot.kde(ax=ax, label='Train', alpha=0.7)
        te_vals.plot.kde(ax=ax, label='Test', alpha=0.7)
    except Exception:
        ax.hist(tr_vals, bins=50, alpha=0.5, label='Train', density=True)
        ax.hist(te_vals, bins=50, alpha=0.5, label='Test', density=True)
    ks_val = ks_df[ks_df['feature'] == c]['ks_stat'].values[0]
    ax.set_title(f'{c}\nKS={ks_val:.4f}', fontsize=10)
    ax.legend()

plt.tight_layout()
plt.savefig('output/eda_deep/eda_02_train_test_kde_top10.png', dpi=100)
plt.close()

print("Section 2 done.", flush=True)

# ============================================================
# Section 3: time_idx Distribution Analysis
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 3: time_idx Distribution Analysis", flush=True)
print("=" * 60, flush=True)

# Check if time_idx exists; some datasets use shift_hour or day_of_week as time proxy
time_col = None
for candidate in ['time_idx', 'shift_hour', 'day_of_week']:
    if candidate in train_df.columns:
        time_col = candidate
        break

if time_col is None:
    print("WARNING: No time_idx column found. Skipping Section 3.", flush=True)
    # Create a placeholder
    with open('output/eda_deep/eda_03_time_idx_stats.txt', 'w') as f:
        f.write("No time_idx column found in the dataset.\n")
        f.write("Available columns that might serve as time proxy:\n")
        f.write(f"  shift_hour present: {'shift_hour' in train_df.columns}\n")
        f.write(f"  day_of_week present: {'day_of_week' in train_df.columns}\n")
else:
    # Check if scenario_id has a time component
    # Assuming each scenario has 25 rows = 25 time steps
    # We can create time_idx from row position within scenario
    pass

# Even without explicit time_idx, analyze shift_hour and day_of_week
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Section 3: Temporal Distribution Analysis', fontsize=16)

# shift_hour analysis
if 'shift_hour' in train_df.columns:
    # Train shift_hour distribution
    sh_stats = train_df.groupby('shift_hour')[TARGET].agg(['mean', 'median', 'std', 'count'])
    axes[0, 0].bar(sh_stats.index, sh_stats['count'], alpha=0.7)
    axes[0, 0].set_title('Train: Row count by shift_hour')
    axes[0, 0].set_xlabel('shift_hour')

    # Target by shift_hour
    axes[0, 1].plot(sh_stats.index, sh_stats['mean'], 'o-', label='Mean')
    axes[0, 1].fill_between(sh_stats.index,
                            sh_stats['mean'] - sh_stats['std'],
                            sh_stats['mean'] + sh_stats['std'],
                            alpha=0.2)
    axes[0, 1].set_title('Target Mean +/- Std by shift_hour')
    axes[0, 1].legend()

    # Train vs Test shift_hour
    if 'shift_hour' in test_df.columns:
        tr_sh = train_df['shift_hour'].value_counts().sort_index()
        te_sh = test_df['shift_hour'].value_counts().sort_index()
        axes[0, 2].bar(tr_sh.index - 0.2, tr_sh.values / tr_sh.sum(), 0.4, label='Train', alpha=0.7)
        axes[0, 2].bar(te_sh.index + 0.2, te_sh.values / te_sh.sum(), 0.4, label='Test', alpha=0.7)
        axes[0, 2].set_title('shift_hour distribution (Train vs Test)')
        axes[0, 2].legend()

# day_of_week analysis
if 'day_of_week' in train_df.columns:
    dow_stats = train_df.groupby('day_of_week')[TARGET].agg(['mean', 'median', 'std', 'count'])
    axes[1, 0].bar(dow_stats.index, dow_stats['count'], alpha=0.7, color='green')
    axes[1, 0].set_title('Train: Row count by day_of_week')

    axes[1, 1].plot(dow_stats.index, dow_stats['mean'], 'o-', color='green', label='Mean')
    axes[1, 1].fill_between(dow_stats.index,
                            dow_stats['mean'] - dow_stats['std'],
                            dow_stats['mean'] + dow_stats['std'],
                            alpha=0.2, color='green')
    axes[1, 1].set_title('Target Mean +/- Std by day_of_week')
    axes[1, 1].legend()

    if 'day_of_week' in test_df.columns:
        tr_dow = train_df['day_of_week'].value_counts().sort_index()
        te_dow = test_df['day_of_week'].value_counts().sort_index()
        axes[1, 2].bar(tr_dow.index - 0.2, tr_dow.values / tr_dow.sum(), 0.4, label='Train', alpha=0.7, color='green')
        axes[1, 2].bar(te_dow.index + 0.2, te_dow.values / te_dow.sum(), 0.4, label='Test', alpha=0.7, color='orange')
        axes[1, 2].set_title('day_of_week distribution (Train vs Test)')
        axes[1, 2].legend()

# Create within-scenario time index
print("  Creating within-scenario time index...", flush=True)
train_df['time_step'] = train_df.groupby('scenario_id').cumcount()
test_df['time_step'] = test_df.groupby('scenario_id').cumcount()

plt.tight_layout()
plt.savefig('output/eda_deep/eda_03_time_idx.png', dpi=100)
plt.close()

# Additional: time_step (within scenario) analysis
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Within-Scenario Time Step Analysis', fontsize=14)

ts_stats = train_df.groupby('time_step')[TARGET].agg(['mean', 'median', 'std', 'count'])
axes2[0].plot(ts_stats.index, ts_stats['mean'], 'o-', label='Mean')
axes2[0].fill_between(ts_stats.index,
                      ts_stats['mean'] - ts_stats['std'],
                      ts_stats['mean'] + ts_stats['std'], alpha=0.2)
axes2[0].set_title('Target by time_step (within scenario)')
axes2[0].legend()

axes2[1].bar(ts_stats.index, ts_stats['count'])
axes2[1].set_title('Row count by time_step')

# time_step vs layout_id heatmap (sample top layouts)
top_layouts = train_df['layout_id'].value_counts().head(15).index
heatmap_data = train_df[train_df['layout_id'].isin(top_layouts)].pivot_table(
    values=TARGET, index='layout_id', columns='time_step', aggfunc='mean')
im = axes2[2].imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd')
axes2[2].set_title('Target: layout_id x time_step')
axes2[2].set_xlabel('time_step')
axes2[2].set_ylabel('layout_id (top 15)')
axes2[2].set_yticks(range(len(heatmap_data.index)))
axes2[2].set_yticklabels(heatmap_data.index, fontsize=7)
plt.colorbar(im, ax=axes2[2])

plt.tight_layout()
plt.savefig('output/eda_deep/eda_03_time_step.png', dpi=100)
plt.close()

# Stats file
time_stats_text = []
time_stats_text.append("=== Temporal Analysis ===")
if 'shift_hour' in train_df.columns:
    time_stats_text.append(f"\nshift_hour unique (train): {train_df['shift_hour'].nunique()}")
    time_stats_text.append(f"shift_hour unique (test): {test_df['shift_hour'].nunique()}")
    time_stats_text.append(f"shift_hour range (train): {train_df['shift_hour'].min()} - {train_df['shift_hour'].max()}")
    time_stats_text.append(f"shift_hour range (test): {test_df['shift_hour'].min()} - {test_df['shift_hour'].max()}")
if 'day_of_week' in train_df.columns:
    time_stats_text.append(f"\nday_of_week unique (train): {train_df['day_of_week'].nunique()}")
    time_stats_text.append(f"day_of_week unique (test): {test_df['day_of_week'].nunique()}")

time_stats_text.append(f"\n=== Within-Scenario Time Step ===")
time_stats_text.append(f"Max time_step (train): {train_df['time_step'].max()}")
time_stats_text.append(f"Max time_step (test): {test_df['time_step'].max()}")
rows_per_scenario_train = train_df.groupby('scenario_id').size()
rows_per_scenario_test = test_df.groupby('scenario_id').size()
time_stats_text.append(f"Rows per scenario (train): min={rows_per_scenario_train.min()}, max={rows_per_scenario_train.max()}, mean={rows_per_scenario_train.mean():.1f}")
time_stats_text.append(f"Rows per scenario (test): min={rows_per_scenario_test.min()}, max={rows_per_scenario_test.max()}, mean={rows_per_scenario_test.mean():.1f}")
time_stats_text.append(f"\nTarget by time_step:\n{ts_stats.to_string()}")

with open('output/eda_deep/eda_03_time_idx_stats.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(time_stats_text))

print("Section 3 done.", flush=True)

# ============================================================
# Section 4: scenario_id Analysis
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 4: scenario_id Analysis", flush=True)
print("=" * 60, flush=True)

train_scenarios = set(train_df['scenario_id'].unique())
test_scenarios = set(test_df['scenario_id'].unique())
overlap_scenarios = train_scenarios & test_scenarios

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Section 4: scenario_id Analysis', fontsize=16)

# Rows per scenario
axes[0, 0].hist(rows_per_scenario_train, bins=50, alpha=0.7, label='Train')
axes[0, 0].hist(rows_per_scenario_test, bins=50, alpha=0.7, label='Test')
axes[0, 0].set_title('Rows per scenario')
axes[0, 0].legend()

# Scenario target mean distribution
scenario_target_mean = train_df.groupby('scenario_id')[TARGET].mean()
axes[0, 1].hist(scenario_target_mean, bins=100, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribution of scenario-level target mean')

# Scenario target std distribution
scenario_target_std = train_df.groupby('scenario_id')[TARGET].std()
axes[0, 2].hist(scenario_target_std.dropna(), bins=100, edgecolor='black', alpha=0.7, color='orange')
axes[0, 2].set_title('Distribution of scenario-level target std')

# Within-scenario variance vs mean
axes[1, 0].scatter(scenario_target_mean, scenario_target_std, alpha=0.3, s=5)
axes[1, 0].set_xlabel('Scenario target mean')
axes[1, 0].set_ylabel('Scenario target std')
axes[1, 0].set_title('Within-scenario: Mean vs Std')

# Check if rows_per_scenario is always 25
exact_25_train = (rows_per_scenario_train == 25).mean() * 100
exact_25_test = (rows_per_scenario_test == 25).mean() * 100
axes[1, 1].bar(['Train (==25)', 'Test (==25)'], [exact_25_train, exact_25_test], color=['blue', 'orange'])
axes[1, 1].set_title(f'% scenarios with exactly 25 rows')
axes[1, 1].set_ylabel('Percentage')
for i, v in enumerate([exact_25_train, exact_25_test]):
    axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center')

# Scenario overlap Venn-like info
axes[1, 2].text(0.5, 0.7, f'Train scenarios: {len(train_scenarios)}', ha='center', fontsize=14, transform=axes[1, 2].transAxes)
axes[1, 2].text(0.5, 0.5, f'Test scenarios: {len(test_scenarios)}', ha='center', fontsize=14, transform=axes[1, 2].transAxes)
axes[1, 2].text(0.5, 0.3, f'Overlap: {len(overlap_scenarios)}', ha='center', fontsize=14, color='red', transform=axes[1, 2].transAxes)
axes[1, 2].set_title('Scenario Overlap')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('output/eda_deep/eda_04_scenario.png', dpi=100)
plt.close()

scenario_text = []
scenario_text.append("=== scenario_id Analysis ===")
scenario_text.append(f"Train unique scenarios: {len(train_scenarios)}")
scenario_text.append(f"Test unique scenarios: {len(test_scenarios)}")
scenario_text.append(f"Overlap scenarios: {len(overlap_scenarios)}")
scenario_text.append(f"Train-only scenarios: {len(train_scenarios - test_scenarios)}")
scenario_text.append(f"Test-only scenarios: {len(test_scenarios - train_scenarios)}")
scenario_text.append(f"\nRows per scenario (train): {rows_per_scenario_train.describe().to_string()}")
scenario_text.append(f"\nRows per scenario (test): {rows_per_scenario_test.describe().to_string()}")
scenario_text.append(f"\n% exactly 25 rows (train): {exact_25_train:.1f}%")
scenario_text.append(f"% exactly 25 rows (test): {exact_25_test:.1f}%")
scenario_text.append(f"\nScenario target mean stats:\n{scenario_target_mean.describe().to_string()}")
scenario_text.append(f"\nScenario target std stats:\n{scenario_target_std.describe().to_string()}")

with open('output/eda_deep/eda_04_scenario_stats.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(scenario_text))

print("Section 4 done.", flush=True)

# ============================================================
# Section 5: layout_id Analysis
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 5: layout_id Analysis", flush=True)
print("=" * 60, flush=True)

train_layouts = set(train_df['layout_id'].unique())
test_layouts = set(test_df['layout_id'].unique())
overlap_layouts = train_layouts & test_layouts

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Section 5: layout_id Analysis', fontsize=16)

# Layout target mean
layout_target = train_df.groupby('layout_id')[TARGET].agg(['mean', 'std', 'count']).sort_values('mean')
axes[0, 0].barh(range(min(30, len(layout_target))),
                layout_target['mean'].values[:30])
axes[0, 0].set_title('Target mean by layout_id (bottom 30)')
axes[0, 0].set_yticks(range(min(30, len(layout_target))))
axes[0, 0].set_yticklabels(layout_target.index[:30], fontsize=6)

# Layout row count
layout_count = train_df['layout_id'].value_counts()
axes[0, 1].hist(layout_count, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Rows per layout_id (train)')

# Layout overlap
axes[0, 2].text(0.5, 0.7, f'Train layouts: {len(train_layouts)}', ha='center', fontsize=14, transform=axes[0, 2].transAxes)
axes[0, 2].text(0.5, 0.5, f'Test layouts: {len(test_layouts)}', ha='center', fontsize=14, transform=axes[0, 2].transAxes)
axes[0, 2].text(0.5, 0.3, f'Overlap: {len(overlap_layouts)}', ha='center', fontsize=14, color='red', transform=axes[0, 2].transAxes)
axes[0, 2].set_title('Layout Overlap')
axes[0, 2].axis('off')

# layout_type boxplot
if 'layout_type' in train_df.columns:
    layout_types = train_df['layout_type'].unique()
    data_by_type = [train_df[train_df['layout_type'] == lt][TARGET].values for lt in layout_types]
    axes[1, 0].boxplot(data_by_type, labels=layout_types)
    axes[1, 0].set_title('Target by layout_type')
    axes[1, 0].tick_params(axis='x', rotation=45)

# layout_info columns correlation with target
layout_num_cols = [c for c in layout_df.columns if c != 'layout_id' and c != 'layout_type'
                   and train_df[c].dtype in ['float64', 'int64']]
if layout_num_cols:
    corrs = train_df[layout_num_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    axes[1, 1].barh(corrs.index, corrs.values)
    axes[1, 1].set_title('layout_info cols vs Target (Pearson)')
    axes[1, 1].tick_params(axis='y', labelsize=7)

# Layout std
axes[1, 2].scatter(layout_target['mean'], layout_target['std'], alpha=0.5, s=20)
axes[1, 2].set_xlabel('Layout target mean')
axes[1, 2].set_ylabel('Layout target std')
axes[1, 2].set_title('Layout: Mean vs Std')

plt.tight_layout()
plt.savefig('output/eda_deep/eda_05_layout.png', dpi=100)
plt.close()

layout_text = []
layout_text.append("=== layout_id Analysis ===")
layout_text.append(f"Train unique layouts: {len(train_layouts)}")
layout_text.append(f"Test unique layouts: {len(test_layouts)}")
layout_text.append(f"Overlap: {len(overlap_layouts)}")
layout_text.append(f"Train-only: {len(train_layouts - test_layouts)}")
layout_text.append(f"Test-only: {len(test_layouts - train_layouts)}")
layout_text.append(f"\nLayout target stats:\n{layout_target.describe().to_string()}")
if layout_num_cols:
    layout_text.append(f"\nLayout info correlations with target:")
    for c in corrs.index:
        layout_text.append(f"  {c}: {corrs[c]:.4f}")

with open('output/eda_deep/eda_05_layout_stats.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(layout_text))

print("Section 5 done.", flush=True)

# ============================================================
# Section 6: Adversarial Validation
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 6: Adversarial Validation", flush=True)
print("=" * 60, flush=True)

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Prepare data - use only numeric columns, exclude IDs
adv_cols = [c for c in num_cols if c in test_df.columns]
X_train_adv = train_df[adv_cols].copy()
X_test_adv = test_df[adv_cols].copy()

X_all = pd.concat([X_train_adv, X_test_adv], axis=0).reset_index(drop=True)
y_adv = np.array([0] * len(X_train_adv) + [1] * len(X_test_adv))

# Fill NaN for LGB
X_all = X_all.fillna(-999)

# 5-fold adversarial validation
print("  Running 5-fold adversarial validation...", flush=True)
oof_adv = np.zeros(len(X_all))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_adv)):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            num_leaves=63, verbosity=-1)
    m.fit(X_all.iloc[tr_idx], y_adv[tr_idx])
    oof_adv[va_idx] = m.predict_proba(X_all.iloc[va_idx])[:, 1]
    fold_auc = roc_auc_score(y_adv[va_idx], oof_adv[va_idx])
    print(f"  Fold {fold+1} AUC: {fold_auc:.4f}", flush=True)

overall_auc = roc_auc_score(y_adv, oof_adv)
print(f"  Overall Adversarial AUC: {overall_auc:.4f}", flush=True)

# Full model for feature importance
m_full = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             num_leaves=63, verbosity=-1)
m_full.fit(X_all, y_adv)
imp_df = pd.DataFrame({
    'feature': adv_cols,
    'importance': m_full.feature_importances_
}).sort_values('importance', ascending=False)

# Now exclude ID-like columns and re-run
exclude_from_adv = ['time_step']  # any proxy for ID/time
adv_cols_clean = [c for c in adv_cols if c not in exclude_from_adv]
X_all_clean = pd.concat([train_df[adv_cols_clean], test_df[adv_cols_clean]], axis=0).reset_index(drop=True).fillna(-999)

print("  Running adversarial validation without ID proxies...", flush=True)
oof_adv_clean = np.zeros(len(X_all_clean))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all_clean, y_adv)):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            num_leaves=63, verbosity=-1)
    m.fit(X_all_clean.iloc[tr_idx], y_adv[tr_idx])
    oof_adv_clean[va_idx] = m.predict_proba(X_all_clean.iloc[va_idx])[:, 1]

clean_auc = roc_auc_score(y_adv, oof_adv_clean)
print(f"  Clean Adversarial AUC (no ID proxies): {clean_auc:.4f}", flush=True)

# Plots
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Section 6: Adversarial Validation Feature Importance', fontsize=16)

# Top 30 feature importance
top30 = imp_df.head(30)
axes[0].barh(range(len(top30)), top30['importance'].values)
axes[0].set_yticks(range(len(top30)))
axes[0].set_yticklabels(top30['feature'].values, fontsize=8)
axes[0].set_title(f'Top 30 Adversarial Features (AUC={overall_auc:.4f})')
axes[0].invert_yaxis()

# Clean model importance
m_clean = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                              num_leaves=63, verbosity=-1)
m_clean.fit(X_all_clean, y_adv)
imp_clean_df = pd.DataFrame({
    'feature': adv_cols_clean,
    'importance': m_clean.feature_importances_
}).sort_values('importance', ascending=False)

top30_clean = imp_clean_df.head(30)
axes[1].barh(range(len(top30_clean)), top30_clean['importance'].values)
axes[1].set_yticks(range(len(top30_clean)))
axes[1].set_yticklabels(top30_clean['feature'].values, fontsize=8)
axes[1].set_title(f'Top 30 (Clean, AUC={clean_auc:.4f})')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('output/eda_deep/eda_06_adv_feature_importance.png', dpi=100)
plt.close()

# Top 10 adversarial features distribution
top10_adv = imp_df.head(10)['feature'].tolist()
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()
fig.suptitle('Top 10 Adversarial Features: Train vs Test Distribution', fontsize=16)

for idx, c in enumerate(top10_adv):
    ax = axes[idx]
    tr_v = train_df[c].dropna()
    te_v = test_df[c].dropna()
    try:
        tr_v.plot.kde(ax=ax, label='Train', alpha=0.7)
        te_v.plot.kde(ax=ax, label='Test', alpha=0.7)
    except Exception:
        ax.hist(tr_v, bins=50, alpha=0.5, label='Train', density=True)
        ax.hist(te_v, bins=50, alpha=0.5, label='Test', density=True)
    ax.set_title(c, fontsize=10)
    ax.legend()

plt.tight_layout()
plt.savefig('output/eda_deep/eda_06_adv_top_features_dist.png', dpi=100)
plt.close()

# Correlation among top adversarial features
top20_adv_feats = imp_df.head(20)['feature'].tolist()
adv_corr = train_df[top20_adv_feats].corr()

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(adv_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(top20_adv_feats)))
ax.set_yticks(range(len(top20_adv_feats)))
ax.set_xticklabels(top20_adv_feats, rotation=90, fontsize=7)
ax.set_yticklabels(top20_adv_feats, fontsize=7)
ax.set_title('Correlation among Top 20 Adversarial Features')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('output/eda_deep/eda_06_adv_correlation.png', dpi=100)
plt.close()

# Save analysis text
adv_text = []
adv_text.append("=== Adversarial Validation Analysis ===")
adv_text.append(f"Overall Adversarial AUC (all features): {overall_auc:.4f}")
adv_text.append(f"Clean Adversarial AUC (no ID proxies): {clean_auc:.4f}")
adv_text.append(f"\nTop 30 Features (all):")
adv_text.append(imp_df.head(30).to_string(index=False))
adv_text.append(f"\nTop 30 Features (clean):")
adv_text.append(imp_clean_df.head(30).to_string(index=False))
adv_text.append(f"\nCorrelation clusters among top adversarial features:")
# Find highly correlated pairs
for i in range(len(top20_adv_feats)):
    for j in range(i+1, len(top20_adv_feats)):
        corr_val = adv_corr.iloc[i, j]
        if abs(corr_val) > 0.7:
            adv_text.append(f"  {top20_adv_feats[i]} <-> {top20_adv_feats[j]}: {corr_val:.3f}")

with open('output/eda_deep/eda_06_adv_auc_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(adv_text))

print("Section 6 done.", flush=True)

# ============================================================
# Section 7: Residual Analysis (Phase 12a)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 7: Residual Analysis", flush=True)
print("=" * 60, flush=True)

import pickle
import glob as glob_module

ckpt_files = glob_module.glob('output/ckpt_phase12a_*.pkl')
has_checkpoint = len(ckpt_files) > 0

if has_checkpoint:
    print(f"  Found checkpoint: {ckpt_files[0]}", flush=True)
    with open(ckpt_files[0], 'rb') as f:
        ckpt = pickle.load(f)
    oof_pred = ckpt['oof']
else:
    print("  No Phase 12a checkpoint found. Using a simple LGB model for residual analysis.", flush=True)
    # Train a quick LGB model to get OOF predictions
    from sklearn.model_selection import KFold
    target_vals = train_df[TARGET].values
    X_train_res = train_df[adv_cols].fillna(-999)
    oof_pred = np.zeros(len(train_df))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_res)):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                               num_leaves=63, verbosity=-1)
        m.fit(X_train_res.iloc[tr_idx], target_vals[tr_idx],
              eval_set=[(X_train_res.iloc[va_idx], target_vals[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_pred[va_idx] = m.predict(X_train_res.iloc[va_idx])
        fold_mae = np.mean(np.abs(target_vals[va_idx] - oof_pred[va_idx]))
        print(f"  Fold {fold+1} MAE: {fold_mae:.4f}", flush=True)

target_vals = train_df[TARGET].values
residual = target_vals - oof_pred
abs_residual = np.abs(residual)
overall_mae = np.mean(abs_residual)
print(f"  Overall MAE: {overall_mae:.4f}", flush=True)

# Top 5% and Bottom 5% error samples
n_samples = len(abs_residual)
top5pct_idx = np.argsort(abs_residual)[-int(n_samples * 0.05):]
bot5pct_idx = np.argsort(abs_residual)[:int(n_samples * 0.05)]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Section 7: Residual Analysis', fontsize=16)

# Residual histogram
axes[0, 0].hist(residual, bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0, color='red', linestyle='--')
axes[0, 0].set_title(f'Residual Distribution (MAE={overall_mae:.4f})')

# Q-Q plot
stats.probplot(residual, plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot of Residuals')

# Residual vs true target (heteroscedasticity)
sample_idx = np.random.choice(len(residual), min(10000, len(residual)), replace=False)
axes[0, 2].scatter(target_vals[sample_idx], residual[sample_idx], alpha=0.1, s=3)
axes[0, 2].axhline(0, color='red', linestyle='--')
axes[0, 2].set_xlabel('True Target')
axes[0, 2].set_ylabel('Residual')
axes[0, 2].set_title('Residual vs True Target')

# Residual by shift_hour
if 'shift_hour' in train_df.columns:
    train_df['_residual'] = residual
    res_by_hour = train_df.groupby('shift_hour')['_residual'].agg(['mean', 'std'])
    axes[1, 0].bar(res_by_hour.index, res_by_hour['mean'], yerr=res_by_hour['std'], alpha=0.7, capsize=3)
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_title('Mean Residual by shift_hour')

# Residual by layout_id (top 20)
train_df['_abs_residual'] = abs_residual
layout_mae = train_df.groupby('layout_id')['_abs_residual'].mean().sort_values(ascending=False)
top20_layouts = layout_mae.head(20)
axes[1, 1].barh(range(len(top20_layouts)), top20_layouts.values)
axes[1, 1].set_yticks(range(len(top20_layouts)))
axes[1, 1].set_yticklabels(top20_layouts.index, fontsize=7)
axes[1, 1].set_title('MAE by layout_id (worst 20)')
axes[1, 1].invert_yaxis()

# Abs residual vs target (where are big errors?)
axes[1, 2].scatter(target_vals[sample_idx], abs_residual[sample_idx], alpha=0.1, s=3)
axes[1, 2].set_xlabel('True Target')
axes[1, 2].set_ylabel('|Residual|')
axes[1, 2].set_title('Abs Residual vs True Target')

plt.tight_layout()
plt.savefig('output/eda_deep/eda_07_residual_analysis.png', dpi=100)
plt.close()

# KS test: top 5% error vs bottom 5% error
ks_residual = []
for c in adv_cols:
    top_vals = train_df.iloc[top5pct_idx][c].dropna().values
    bot_vals = train_df.iloc[bot5pct_idx][c].dropna().values
    if len(top_vals) > 10 and len(bot_vals) > 10:
        stat, pval = stats.ks_2samp(top_vals, bot_vals)
        ks_residual.append({'feature': c, 'ks_stat': stat, 'p_value': pval,
                           'top5_mean': np.mean(top_vals), 'bot5_mean': np.mean(bot_vals)})

ks_res_df = pd.DataFrame(ks_residual).sort_values('ks_stat', ascending=False)

residual_text = []
residual_text.append("=== Residual Analysis ===")
residual_text.append(f"Checkpoint used: {'Phase 12a' if has_checkpoint else 'Quick LGB (no checkpoint found)'}")
residual_text.append(f"Overall MAE: {overall_mae:.4f}")
residual_text.append(f"Residual mean: {residual.mean():.4f}")
residual_text.append(f"Residual std: {residual.std():.4f}")
residual_text.append(f"Residual skewness: {pd.Series(residual).skew():.4f}")
residual_text.append(f"\n=== Top 5% Error vs Bottom 5% Error (KS Test) ===")
residual_text.append(f"Top 5% error threshold: {np.sort(abs_residual)[-int(n_samples*0.05)]:.4f}")
residual_text.append(f"\nTop 20 most different features between high-error and low-error samples:")
residual_text.append(ks_res_df.head(20).to_string(index=False))

# Characteristics of high-error samples
residual_text.append(f"\n=== High-Error Sample Characteristics ===")
residual_text.append(f"Target mean (top 5% error): {target_vals[top5pct_idx].mean():.4f}")
residual_text.append(f"Target mean (bottom 5% error): {target_vals[bot5pct_idx].mean():.4f}")
residual_text.append(f"Target mean (all): {target_vals.mean():.4f}")

if 'layout_type' in train_df.columns:
    residual_text.append(f"\nLayout type distribution (top 5% error):")
    top5_lt = train_df.iloc[top5pct_idx]['layout_type'].value_counts(normalize=True)
    all_lt = train_df['layout_type'].value_counts(normalize=True)
    for lt in all_lt.index:
        t5 = top5_lt.get(lt, 0)
        residual_text.append(f"  {lt}: top5%={t5:.3f}, all={all_lt[lt]:.3f}")

# Clean up temp columns
train_df.drop(['_residual', '_abs_residual'], axis=1, inplace=True, errors='ignore')

with open('output/eda_deep/eda_07_residual_top_errors.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(residual_text))

print("Section 7 done.", flush=True)

# ============================================================
# Section 8: Target-Feature Correlation
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 8: Target-Feature Correlation", flush=True)
print("=" * 60, flush=True)

# Pearson correlation
pearson_corr = train_df[adv_cols + [TARGET]].corr(method='pearson')[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

# Spearman correlation
spearman_corr = train_df[adv_cols + [TARGET]].corr(method='spearman')[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

# Mutual Information (sample for speed)
print("  Computing Mutual Information (sampled)...", flush=True)
mi_sample_idx = np.random.choice(len(train_df), min(50000, len(train_df)), replace=False)
X_mi = train_df.iloc[mi_sample_idx][adv_cols].fillna(-999)
y_mi = train_df.iloc[mi_sample_idx][TARGET].values
mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': adv_cols, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)

# Find features with weak linear but strong nonlinear
pearson_abs = pearson_corr.abs()
mi_norm = mi_df.set_index('feature')['mi_score']
mi_norm = mi_norm / mi_norm.max()  # normalize to [0, 1]
pearson_norm = pearson_abs / pearson_abs.max()

nonlinear_candidates = []
for feat in adv_cols:
    if feat in pearson_norm.index and feat in mi_norm.index:
        p = pearson_norm[feat]
        m = mi_norm[feat]
        if m > 0.3 and p < 0.15:  # strong MI but weak Pearson
            nonlinear_candidates.append({'feature': feat, 'pearson_abs': pearson_abs.get(feat, 0),
                                        'mi_score': mi_df[mi_df['feature'] == feat]['mi_score'].values[0]})

nonlinear_df = pd.DataFrame(nonlinear_candidates).sort_values('mi_score', ascending=False)

# Plots
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Section 8: Feature-Target Correlation', fontsize=16)

# Pearson Top 30
top30_pearson = pearson_corr.head(30)
axes[0, 0].barh(range(len(top30_pearson)), top30_pearson.values)
axes[0, 0].set_yticks(range(len(top30_pearson)))
axes[0, 0].set_yticklabels(top30_pearson.index, fontsize=7)
axes[0, 0].set_title('Top 30 Pearson Correlation with Target')
axes[0, 0].invert_yaxis()

# Spearman Top 30
top30_spearman = spearman_corr.head(30)
axes[0, 1].barh(range(len(top30_spearman)), top30_spearman.values)
axes[0, 1].set_yticks(range(len(top30_spearman)))
axes[0, 1].set_yticklabels(top30_spearman.index, fontsize=7)
axes[0, 1].set_title('Top 30 Spearman Correlation with Target')
axes[0, 1].invert_yaxis()

# MI Top 30
top30_mi = mi_df.head(30)
axes[1, 0].barh(range(len(top30_mi)), top30_mi['mi_score'].values)
axes[1, 0].set_yticks(range(len(top30_mi)))
axes[1, 0].set_yticklabels(top30_mi['feature'].values, fontsize=7)
axes[1, 0].set_title('Top 30 Mutual Information with Target')
axes[1, 0].invert_yaxis()

# Heatmap of top 30 pearson features
top30_feats = pearson_corr.head(30).index.tolist()
heatmap_corr = train_df[top30_feats].corr()
im = axes[1, 1].imshow(heatmap_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(top30_feats)))
axes[1, 1].set_yticks(range(len(top30_feats)))
axes[1, 1].set_xticklabels(top30_feats, rotation=90, fontsize=5)
axes[1, 1].set_yticklabels(top30_feats, fontsize=5)
axes[1, 1].set_title('Correlation Heatmap (Top 30)')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('output/eda_deep/eda_08_correlation_top30.png', dpi=100)
plt.close()

corr_text = []
corr_text.append("=== Feature-Target Correlation ===")
corr_text.append(f"\nTop 30 Pearson:")
for feat, val in pearson_corr.head(30).items():
    corr_text.append(f"  {feat}: {val:.4f}")
corr_text.append(f"\nTop 30 Spearman:")
for feat, val in spearman_corr.head(30).items():
    corr_text.append(f"  {feat}: {val:.4f}")
corr_text.append(f"\nTop 30 Mutual Information:")
for _, row in mi_df.head(30).iterrows():
    corr_text.append(f"  {row['feature']}: {row['mi_score']:.4f}")
corr_text.append(f"\n=== Nonlinear Candidates (strong MI, weak Pearson) ===")
if len(nonlinear_df) > 0:
    corr_text.append(nonlinear_df.to_string(index=False))
else:
    corr_text.append("  None found with MI_norm > 0.3 and Pearson < 0.15")

with open('output/eda_deep/eda_08_feature_target_correlation.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(corr_text))

print("Section 8 done.", flush=True)

# ============================================================
# Section 9: Dynamic Column Temporal Patterns
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 9: Dynamic Column Temporal Patterns", flush=True)
print("=" * 60, flush=True)

dynamic_cols = ['robot_active', 'robot_charging', 'charge_queue_length',
                'pack_utilization', 'congestion_score', 'order_inflow_15m']
dynamic_cols = [c for c in dynamic_cols if c in train_df.columns]

# Sample 10 scenarios
sample_scenarios = train_df['scenario_id'].unique()[:10]

fig, axes = plt.subplots(len(dynamic_cols), 2, figsize=(18, 5 * len(dynamic_cols)))
if len(dynamic_cols) == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Section 9: Dynamic Column Patterns', fontsize=16, y=1.01)

for row_idx, col in enumerate(dynamic_cols):
    # Left: sample scenario traces
    ax_left = axes[row_idx, 0]
    for sc in sample_scenarios:
        sc_data = train_df[train_df['scenario_id'] == sc].sort_values('time_step')
        ax_left.plot(sc_data['time_step'].values, sc_data[col].values, alpha=0.5, linewidth=1)
    ax_left.set_title(f'{col} - Sample Scenarios')
    ax_left.set_xlabel('time_step')
    ax_left.set_ylabel(col)

    # Right: global mean +/- std
    ax_right = axes[row_idx, 1]
    global_stats = train_df.groupby('time_step')[col].agg(['mean', 'std'])
    ax_right.plot(global_stats.index, global_stats['mean'], 'b-', linewidth=2, label='Mean')
    ax_right.fill_between(global_stats.index,
                          global_stats['mean'] - global_stats['std'],
                          global_stats['mean'] + global_stats['std'],
                          alpha=0.2, color='blue')
    ax_right.set_title(f'{col} - Global Mean +/- Std')
    ax_right.set_xlabel('time_step')
    ax_right.legend()

plt.tight_layout()
plt.savefig('output/eda_deep/eda_09_dynamic_patterns.png', dpi=100)
plt.close()

print("Section 9 done.", flush=True)

# ============================================================
# Section 10: Final Summary (eda_summary.md)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Section 10: Generating Summary", flush=True)
print("=" * 60, flush=True)

summary_lines = []
summary_lines.append("# EDA Deep Analysis Summary")
summary_lines.append("")

# 1. Target
summary_lines.append("## 1. Target Distribution")
summary_lines.append(f"- Mean: {target.mean():.4f}, Std: {target.std():.4f}")
summary_lines.append(f"- Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")
summary_lines.append(f"- Zero ratio: {zero_pct:.2f}%")
summary_lines.append(f"- Outlier ratio (IQR): {outlier_pct:.2f}%")
summary_lines.append(f"- Q50: {quantiles[0.50]:.4f}, Q95: {quantiles[0.95]:.4f}, Q99: {quantiles[0.99]:.4f}")
summary_lines.append(f"- log1p transformation recommended if skewness > 2")
summary_lines.append("")

# 2. Train vs Test
summary_lines.append("## 2. Train vs Test Distribution Difference")
summary_lines.append(f"- Columns with p < 0.05 (KS test): {(ks_df['p_value'] < 0.05).sum()} / {len(ks_df)}")
summary_lines.append(f"- Columns with KS > 0.1: {(ks_df['ks_stat'] > 0.1).sum()}")
summary_lines.append(f"- Top 5 most different columns:")
for _, row in ks_df.head(5).iterrows():
    summary_lines.append(f"  - {row['feature']}: KS={row['ks_stat']:.4f}")
summary_lines.append("")

# 3. Adversarial
summary_lines.append("## 3. Adversarial Validation")
summary_lines.append(f"- Overall Adversarial AUC (all features): {overall_auc:.4f}")
summary_lines.append(f"- Clean Adversarial AUC (no ID proxies): {clean_auc:.4f}")
summary_lines.append(f"- Top 5 distinguishing features:")
for _, row in imp_df.head(5).iterrows():
    summary_lines.append(f"  - {row['feature']}: importance={row['importance']}")
summary_lines.append("")

# 4. Structure
summary_lines.append("## 4. Data Structure")
summary_lines.append(f"- Scenario overlap (train/test): {len(overlap_scenarios)}")
summary_lines.append(f"  - Train scenarios: {len(train_scenarios)}, Test: {len(test_scenarios)}")
summary_lines.append(f"- Layout overlap (train/test): {len(overlap_layouts)}")
summary_lines.append(f"  - Train layouts: {len(train_layouts)}, Test: {len(test_layouts)}")
summary_lines.append(f"- Rows per scenario = 25: Train {exact_25_train:.1f}%, Test {exact_25_test:.1f}%")
summary_lines.append("")

# 5. Residual
summary_lines.append("## 5. Residual Analysis")
summary_lines.append(f"- Overall MAE: {overall_mae:.4f}")
summary_lines.append(f"- Residual bias (mean): {residual.mean():.4f}")
summary_lines.append(f"- Top features different between high-error and low-error:")
for _, row in ks_res_df.head(5).iterrows():
    summary_lines.append(f"  - {row['feature']}: KS={row['ks_stat']:.4f}")
summary_lines.append("")

# 6. CV-Public gap hypotheses
summary_lines.append("## 6. CV-Public Gap Hypotheses")
summary_lines.append("- Hypothesis 1: Train/test come from different scenario distributions")
summary_lines.append(f"  (Evidence: scenario overlap = {len(overlap_scenarios)}, Adversarial AUC = {overall_auc:.4f})")
summary_lines.append("- Hypothesis 2: Time-dependent features shift between train and test periods")
summary_lines.append(f"  (Evidence: Top KS features are time-dependent dynamic columns)")
summary_lines.append("- Hypothesis 3: Overfitting to train-specific scenario/layout patterns")
summary_lines.append(f"  (Evidence: high within-scenario variance, layout-specific biases in residual)")
summary_lines.append("- Hypothesis 4: Feature leakage from ID-encoding columns inflating CV")
summary_lines.append(f"  (Evidence: Adversarial AUC with vs without ID proxies)")
summary_lines.append("")

# 7. Phase 13 Strategy
summary_lines.append("## 7. Phase 13 Strategy Recommendations")
summary_lines.append("")
summary_lines.append("### Features to remove")
summary_lines.append("- Features with KS > 0.2 (high train/test shift) - see Section 2 ranking")
summary_lines.append("- Consider removing top adversarial features if they don't correlate with target")
summary_lines.append("")
summary_lines.append("### Features to add")
summary_lines.append("- Nonlinear interaction features for columns with high MI but low Pearson")
if len(nonlinear_df) > 0:
    for _, row in nonlinear_df.head(5).iterrows():
        summary_lines.append(f"  - {row['feature']}: MI={row['mi_score']:.4f}, |Pearson|={row['pearson_abs']:.4f}")
summary_lines.append("")
summary_lines.append("### Weight adjustment")
summary_lines.append("- Down-weight samples with high adversarial probability (likely non-representative)")
summary_lines.append("")
summary_lines.append("### CV strategy")
summary_lines.append("- Use GroupKFold on scenario_id to avoid data leakage within scenarios")
summary_lines.append("- Consider time-aware splits if temporal structure exists")
summary_lines.append("")
summary_lines.append("### Model architecture")
summary_lines.append("- Focus on robust ensemble rather than complex single models")
summary_lines.append("- Reduce overfitting: fewer leaves, higher min_child_samples, feature subsampling")
summary_lines.append("")
summary_lines.append("---")
summary_lines.append("*Generated by run_eda_deep.py*")

with open('output/eda_deep/eda_summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print("Section 10 done.", flush=True)

# ============================================================
# Final
# ============================================================
print("\n" + "=" * 60, flush=True)
print("ALL SECTIONS COMPLETE", flush=True)
print("Output saved to: output/eda_deep/", flush=True)
print("=" * 60, flush=True)
