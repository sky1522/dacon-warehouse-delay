"""
Phase 16 EDA: Verify scenario time order for lag features
- Is row order within scenario = time order?
- Autocorrelation, lag/diff/rolling correlation with target
- Decision: whether Phase 16 should use lag features
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output/phase16_eda', exist_ok=True)

# ============================================================
# 1. Data Load + Basic Check
# ============================================================
print("=" * 60)
print("=== 1. Data Load + Basic Check ===")
print("=" * 60)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")
print(f"Train scenarios: {train['scenario_id'].nunique()}")
print(f"Test scenarios:  {test['scenario_id'].nunique()}")
print(f"\nRows per scenario (train):")
print(train.groupby('scenario_id').size().describe())
print(f"\nRows per scenario (test):")
print(test.groupby('scenario_id').size().describe())

print(f"\nFirst 10 rows of train[['ID', 'scenario_id']]:")
print(train[['ID', 'scenario_id']].head(10))

# ============================================================
# 2. Row Order Structure
# ============================================================
print("\n" + "=" * 60)
print("=== 2. Row Order Structure ===")
print("=" * 60)

sample_ids = train['ID'].head(30).tolist()
print(f"Sample IDs: {sample_ids[:10]}")

try:
    train['id_num'] = pd.to_numeric(train['ID'], errors='coerce')
    if train['id_num'].notna().all():
        print("ID is numeric")
    else:
        train['id_num'] = train['ID'].astype(str).str.extract(r'(\d+)').astype(int)
        print("ID number extracted from string")

    first_scenario = train['scenario_id'].iloc[0]
    scen0 = train[train['scenario_id'] == first_scenario].head(25)
    print(f"\nFirst scenario ({first_scenario}) IDs in order:")
    print(scen0[['ID', 'id_num']].to_string())

    is_monotonic = scen0['id_num'].is_monotonic_increasing
    print(f"\nFirst scenario IDs monotonic increasing: {is_monotonic}")
except Exception as e:
    print(f"ID parsing failed: {e}")
    train['id_num'] = np.arange(len(train))

train['row_in_scenario'] = train.groupby('scenario_id').cumcount()
print(f"\nrow_in_scenario range: {train['row_in_scenario'].min()} ~ {train['row_in_scenario'].max()}")

# ============================================================
# 3. shift_hour Pattern Within Scenario
# ============================================================
print("\n" + "=" * 60)
print("=== 3. shift_hour Pattern Within Scenario ===")
print("=" * 60)

sample_scenarios = train['scenario_id'].unique()[:5]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, sid in enumerate(sample_scenarios):
    sdf = train[train['scenario_id'] == sid].sort_values('row_in_scenario')
    axes[i].plot(sdf['row_in_scenario'], sdf['shift_hour'], marker='o', markersize=4)
    axes[i].set_title(f'Scenario {sid}')
    axes[i].set_xlabel('row_in_scenario')
    axes[i].set_ylabel('shift_hour')
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/phase16_eda/shift_hour_in_scenario.png', dpi=100)
plt.close()
print("  Saved: shift_hour_in_scenario.png")


def is_non_decreasing(x):
    x_clean = x.dropna()
    if len(x_clean) < 2:
        return True
    return (x_clean.diff().dropna() >= 0).all()


mono_ratio = train.groupby('scenario_id')['shift_hour'].apply(is_non_decreasing).mean()
print(f"\nScenarios with non-decreasing shift_hour: {mono_ratio*100:.1f}%")

if 'day_of_week' in train.columns:
    mono_dow = train.groupby('scenario_id')['day_of_week'].apply(is_non_decreasing).mean()
    print(f"Scenarios with non-decreasing day_of_week: {mono_dow*100:.1f}%")

# Also check: shift_hour constant within scenario? (all same hour?)
const_ratio = train.groupby('scenario_id')['shift_hour'].apply(lambda x: x.nunique() == 1).mean()
print(f"Scenarios with constant shift_hour: {const_ratio*100:.1f}%")

# ============================================================
# 4. Autocorrelation Within Scenario
# ============================================================
print("\n" + "=" * 60)
print("=== 4. Autocorrelation Within Scenario ===")
print("=" * 60)

AUTOCORR_COLS = [
    'congestion_score', 'robot_active', 'pack_utilization',
    'order_inflow_15m', 'charge_queue_length', 'battery_mean',
    'robot_charging', 'max_zone_density', 'unique_sku_15m',
    'outbound_truck_wait_min',
]
AUTOCORR_COLS = [c for c in AUTOCORR_COLS if c in train.columns]


def calc_autocorr(x):
    x_clean = x.dropna()
    if len(x_clean) < 5:
        return np.nan
    return x_clean.autocorr(lag=1)


autocorr_results = {}
for col in AUTOCORR_COLS:
    autocorrs = train.groupby('scenario_id')[col].apply(calc_autocorr).dropna()
    autocorr_results[col] = {
        'mean': autocorrs.mean(),
        'median': autocorrs.median(),
        'std': autocorrs.std(),
        'positive_ratio': (autocorrs > 0.3).mean(),
    }

print("\nLag-1 autocorrelation (scenario-internal):")
print(f"{'Column':<30} {'Mean':<8} {'Median':<8} {'Std':<8} {'% > 0.3':<10}")
print("-" * 70)
for col, stats in autocorr_results.items():
    print(f"{col:<30} {stats['mean']:<8.3f} {stats['median']:<8.3f} "
          f"{stats['std']:<8.3f} {stats['positive_ratio']*100:<10.1f}")

overall_autocorr = np.mean([s['mean'] for s in autocorr_results.values()])
print(f"\nOverall mean autocorr: {overall_autocorr:.3f}")
print("Interpretation:")
print("  > 0.5:  Strong time structure (lag very useful)")
print("  0.3-0.5: Time structure exists (lag useful)")
print("  0.1-0.3: Weak time structure (lag marginal)")
print("  < 0.1:  Random order (lag useless)")

with open('output/phase16_eda/autocorr_results.txt', 'w') as f:
    f.write(f"Overall mean lag-1 autocorr: {overall_autocorr:.4f}\n\n")
    for col, stats in autocorr_results.items():
        f.write(f"{col}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
                f"positive_ratio={stats['positive_ratio']:.3f}\n")

# ============================================================
# 5. Lag Feature vs Target Correlation
# ============================================================
print("\n" + "=" * 60)
print("=== 5. Lag Feature vs Target Correlation ===")
print("=" * 60)

y = train['avg_delay_minutes_next_30m'].values

lag_corr_results = {}
for col in AUTOCORR_COLS:
    for lag in [1, 2, 3]:
        lag_col = f'lag{lag}_{col}'
        train[lag_col] = train.groupby('scenario_id')[col].shift(lag)

        mask = train[lag_col].notna()
        if mask.sum() > 100:
            corr = np.corrcoef(train.loc[mask, lag_col], y[mask])[0, 1]
            lag_corr_results[lag_col] = corr

print(f"\n{'Feature':<35} {'Corr w/ target':<12}")
print("-" * 50)
for col in AUTOCORR_COLS:
    base_corr = np.corrcoef(train[col].fillna(0), y)[0, 1]
    print(f"{col:<35} {base_corr:>10.4f}  (current)")
    for lag in [1, 2, 3]:
        lag_col = f'lag{lag}_{col}'
        if lag_col in lag_corr_results:
            print(f"  {lag_col:<33} {lag_corr_results[lag_col]:>10.4f}")

print("\n\nLag utility score (lag_corr / current_corr):")
print("  1.0+: lag as useful as current")
print("  0.7-1.0: lag useful")
print("  < 0.5: lag marginal")

pd.DataFrame([
    {'feature': col, 'lag_corr': lc}
    for col, lc in lag_corr_results.items()
]).to_csv('output/phase16_eda/lag_corr.csv', index=False)

# ============================================================
# 6. Diff Features
# ============================================================
print("\n" + "=" * 60)
print("=== 6. Diff Feature vs Target Correlation ===")
print("=" * 60)

diff_corr_results = {}
for col in AUTOCORR_COLS:
    diff_col = f'diff1_{col}'
    train[diff_col] = train[col] - train[f'lag1_{col}']

    mask = train[diff_col].notna()
    if mask.sum() > 100:
        corr = np.corrcoef(train.loc[mask, diff_col], y[mask])[0, 1]
        diff_corr_results[diff_col] = corr

print("\nDiff feature correlation with target:")
for col, corr in sorted(diff_corr_results.items(), key=lambda x: abs(x[1]), reverse=True):
    marker = " *" if abs(corr) > 0.05 else ""
    print(f"  {col:<35} {corr:>8.4f}{marker}")

if diff_corr_results:
    strongest_diff = sorted(diff_corr_results.items(), key=lambda x: abs(x[1]), reverse=True)[0]
    print(f"\nStrongest diff feature: {strongest_diff[0]} (corr={strongest_diff[1]:.4f})")

# ============================================================
# 7. Rolling Features
# ============================================================
print("\n" + "=" * 60)
print("=== 7. Rolling Mean vs Target Correlation ===")
print("=" * 60)

rolling_corr_results = {}
for col in AUTOCORR_COLS[:5]:
    for window in [3, 5]:
        roll_col = f'roll{window}_{col}'
        train[roll_col] = train.groupby('scenario_id')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        corr = np.corrcoef(train[roll_col].fillna(0), y)[0, 1]
        rolling_corr_results[roll_col] = corr

print("\nRolling mean correlation with target:")
for col, corr in rolling_corr_results.items():
    print(f"  {col:<35} {corr:>8.4f}")

# ============================================================
# 8. Visualization
# ============================================================
print("\n=== Visualization ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Autocorrelation bar chart
cols_sorted = sorted(autocorr_results.keys(), key=lambda c: autocorr_results[c]['mean'], reverse=True)
axes[0, 0].barh(range(len(cols_sorted)), [autocorr_results[c]['mean'] for c in cols_sorted])
axes[0, 0].set_yticks(range(len(cols_sorted)))
axes[0, 0].set_yticklabels(cols_sorted, fontsize=8)
axes[0, 0].set_xlabel('Mean Lag-1 Autocorrelation')
axes[0, 0].set_title('Autocorrelation by Feature')
axes[0, 0].axvline(0.3, color='red', linestyle='--', alpha=0.5)

# Lag corr vs current corr scatter
current_corrs = {}
for col in AUTOCORR_COLS:
    current_corrs[col] = abs(np.corrcoef(train[col].fillna(0), y)[0, 1])
lag1_corrs = {}
for col in AUTOCORR_COLS:
    k = f'lag1_{col}'
    if k in lag_corr_results:
        lag1_corrs[col] = abs(lag_corr_results[k])

common_cols = [c for c in AUTOCORR_COLS if c in lag1_corrs]
if common_cols:
    axes[0, 1].scatter([current_corrs[c] for c in common_cols],
                       [lag1_corrs[c] for c in common_cols], s=50)
    for c in common_cols:
        axes[0, 1].annotate(c[:15], (current_corrs[c], lag1_corrs[c]), fontsize=6)
    axes[0, 1].plot([0, 0.4], [0, 0.4], 'k--', alpha=0.3)
    axes[0, 1].set_xlabel('|Current corr with target|')
    axes[0, 1].set_ylabel('|Lag-1 corr with target|')
    axes[0, 1].set_title('Current vs Lag-1 Correlation')

# Diff corr bar chart
if diff_corr_results:
    diff_sorted = sorted(diff_corr_results.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [d[0].replace('diff1_', '') for d in diff_sorted]
    vals = [d[1] for d in diff_sorted]
    axes[0, 2].barh(range(len(names)), vals)
    axes[0, 2].set_yticks(range(len(names)))
    axes[0, 2].set_yticklabels(names, fontsize=8)
    axes[0, 2].set_xlabel('Correlation with target')
    axes[0, 2].set_title('Diff Features vs Target')

# Sample scenario traces (3 features)
trace_cols = AUTOCORR_COLS[:3]
sid = train['scenario_id'].iloc[0]
sdf = train[train['scenario_id'] == sid].sort_values('row_in_scenario')
for i, col in enumerate(trace_cols):
    axes[1, i].plot(sdf['row_in_scenario'], sdf[col], marker='o', markersize=4, label=col)
    axes[1, i].set_xlabel('row_in_scenario')
    axes[1, i].set_ylabel(col)
    axes[1, i].set_title(f'{col} trace (scenario {sid})')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/phase16_eda/phase16_analysis.png', dpi=100)
plt.close()
print("  Saved: phase16_analysis.png")

# ============================================================
# 9. Final Conclusion
# ============================================================
print("\n" + "=" * 70)
print("=== FINAL CONCLUSION ===")
print("=" * 70)

print(f"\n1. Row order structure:")
print(f"   - shift_hour non-decreasing ratio: {mono_ratio*100:.1f}%")
print(f"   - shift_hour constant ratio: {const_ratio*100:.1f}%")
print(f"   - Overall autocorr: {overall_autocorr:.3f}")

max_lag_corr = max([abs(v) for v in lag_corr_results.values()]) if lag_corr_results else 0
max_diff_corr = max([abs(v) for v in diff_corr_results.values()]) if diff_corr_results else 0
print(f"\n2. Max lag correlation with target: {max_lag_corr:.4f}")
print(f"3. Max diff correlation with target: {max_diff_corr:.4f}")

print("\n=== Decision Guide ===")
if overall_autocorr > 0.3 and mono_ratio > 0.5:
    decision = "STRONG"
    print("STRONG: scenario time structure exists. Phase 16 lag features recommended")
elif overall_autocorr > 0.1 and max_lag_corr > 0.05:
    decision = "WEAK"
    print("WEAK: weak time structure. Diff/rolling only, skip raw lag")
elif max_lag_corr < 0.03:
    decision = "NONE"
    print("NONE: no time order. Skip lag features entirely")
    print("  -> Focus on interaction + 2nd-order aggregation")
else:
    decision = "AMBIGUOUS"
    print("AMBIGUOUS: try small lag (1, 2 only) + let selection decide")

print("\n=== Phase 16 Recommendation ===")
if overall_autocorr > 0.3:
    print("-> Phase 16 Full: Lag + Diff + Roll + Interaction + 2nd-order agg")
elif overall_autocorr > 0.1:
    print("-> Phase 16 Partial: Diff + Interaction + 2nd-order agg (lag minimal)")
else:
    print("-> Phase 16 Revised: Interaction + 2nd-order agg only")

# Summary file
with open('output/phase16_eda/summary.md', 'w') as f:
    f.write("# Phase 16 EDA Summary\n\n")
    f.write(f"## Time order verification\n")
    f.write(f"- shift_hour monotonic ratio: {mono_ratio*100:.1f}%\n")
    f.write(f"- shift_hour constant ratio: {const_ratio*100:.1f}%\n")
    f.write(f"- Overall lag-1 autocorrelation: {overall_autocorr:.4f}\n\n")
    f.write(f"## Lag feature potential\n")
    f.write(f"- Max lag correlation with target: {max_lag_corr:.4f}\n")
    f.write(f"- Max diff correlation with target: {max_diff_corr:.4f}\n\n")
    f.write(f"## Decision: {decision}\n\n")
    if decision == "STRONG":
        f.write("Phase 16: Full lag/diff/rolling features recommended\n")
    elif decision == "WEAK":
        f.write("Phase 16: Diff/rolling only, skip raw lag\n")
    elif decision == "NONE":
        f.write("Phase 16: Skip lag, focus on interaction/aggregation\n")
    else:
        f.write("Phase 16: Try minimal lag, let selection decide\n")

print("\n=== All results saved to output/phase16_eda/ ===")
print("  - shift_hour_in_scenario.png")
print("  - phase16_analysis.png")
print("  - autocorr_results.txt")
print("  - lag_corr.csv")
print("  - summary.md")
