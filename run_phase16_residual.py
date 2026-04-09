"""
Phase 16 Residual Analysis for Phase 17 Direction
- Phase 16 ensemble OOF residual을 다양한 각도로 분석
- 결과: output/phase16_residual/
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('output/phase16_residual', exist_ok=True)

# ============================================================
# Part 1: Phase 16 OOF 재구성
# ============================================================

model_names = ['lgb_raw', 'lgb_huber', 'lgb_sqrt', 'xgb', 'cat_log1p', 'cat_raw', 'mlp']
p16_weights = {
    'lgb_raw': 0.0086, 'lgb_huber': 0.3226, 'lgb_sqrt': 0.1472,
    'xgb': -0.0752, 'cat_log1p': 0.1776, 'cat_raw': -0.0287, 'mlp': 0.4458,
}

oofs = {}
for name in model_names:
    with open(f'output/ckpt_phase16_{name}.pkl', 'rb') as f:
        ckpt = pickle.load(f)
    oofs[name] = ckpt['oof']

# Weighted ensemble OOF
oof_p16 = np.zeros(len(oofs[model_names[0]]))
for name in model_names:
    oof_p16 += p16_weights[name] * oofs[name]

# Load train data
train_df = pd.read_csv('data/train.csv')
layout_info = pd.read_csv('data/layout_info.csv')
train_df = train_df.merge(layout_info, on='layout_id', how='left')

y_true = train_df['avg_delay_minutes_next_30m'].values

# Residual
residual = y_true - oof_p16
abs_residual = np.abs(residual)

print(f"Phase 16 OOF MAE: {abs_residual.mean():.4f}")
print(f"Residual stats: mean={residual.mean():.3f}, std={residual.std():.3f}")
print(f"Residual range: [{residual.min():.2f}, {residual.max():.2f}]")

# ============================================================
# Part 2: Target bin별 residual (Bin 9 여전히 문제인가?)
# ============================================================

print("\n=== Residual by target bin ===")

# 10 bins
bins = pd.qcut(y_true, q=10, labels=False, duplicates='drop')
bin_stats = pd.DataFrame({
    'y': y_true,
    'oof': oof_p16,
    'residual': residual,
    'abs_residual': abs_residual,
    'bin': bins
})

bin_summary = bin_stats.groupby('bin').agg(
    target_mean=('y', 'mean'),
    target_max=('y', 'max'),
    oof_mean=('oof', 'mean'),
    residual_mean=('residual', 'mean'),  # 양수면 과소예측
    abs_mae=('abs_residual', 'mean'),
    count=('y', 'size')
).round(3)

print(bin_summary.to_string())

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(bin_summary.index, bin_summary['abs_mae'])
axes[0].set_xlabel('Target bin (0=lowest, 9=highest)')
axes[0].set_ylabel('MAE')
axes[0].set_title('Phase 16 MAE by target bin')
axes[0].grid(True, alpha=0.3)

axes[1].bar(bin_summary.index, bin_summary['residual_mean'],
            color=['red' if r > 0 else 'blue' for r in bin_summary['residual_mean']])
axes[1].set_xlabel('Target bin')
axes[1].set_ylabel('Residual mean (positive = underprediction)')
axes[1].set_title('Phase 16 Residual by target bin')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/phase16_residual/residual_by_bin.png', dpi=100)
plt.close()
bin_summary.to_csv('output/phase16_residual/residual_by_bin.csv')

# 핵심 질문: bin 9 residual이 여전히 큰가?
bin9_stats = bin_summary.iloc[-1]
print(f"\n★ Bin 9 (highest 10%):")
print(f"  target_mean: {bin9_stats['target_mean']:.2f}")
print(f"  oof_mean: {bin9_stats['oof_mean']:.2f}")
print(f"  residual: {bin9_stats['residual_mean']:+.2f}")
print(f"  MAE: {bin9_stats['abs_mae']:.2f}")

# ============================================================
# Part 3: Layout별 residual (어떤 layout이 여전히 어려운가?)
# ============================================================

print("\n=== Residual by layout_id ===")

layout_stats = pd.DataFrame({
    'layout_id': train_df['layout_id'].values,
    'y': y_true,
    'oof': oof_p16,
    'abs_residual': abs_residual,
    'residual': residual,
}).groupby('layout_id').agg(
    target_mean=('y', 'mean'),
    oof_mean=('oof', 'mean'),
    mae=('abs_residual', 'mean'),
    residual_mean=('residual', 'mean'),
    count=('y', 'size')
).sort_values('mae', ascending=False)

print(f"Top 10 hardest layouts (Phase 16):")
print(layout_stats.head(10).to_string())

print(f"\nTop 10 easiest layouts (Phase 16):")
print(layout_stats.tail(10).to_string())

# Phase 13s1과 비교 (있으면)
if os.path.exists('output/phase13s2_analysis/layout_mae_ranking.csv'):
    p13_ranking = pd.read_csv('output/phase13s2_analysis/layout_mae_ranking.csv')
    print(f"\nPhase 13s1 baseline hard layouts MAE: {p13_ranking[p13_ranking['difficulty']=='hard']['mae'].mean():.4f}")

    # Phase 16 hard layouts MAE
    hard_ids = p13_ranking[p13_ranking['difficulty']=='hard']['layout_id'].tolist()
    hard_mask = train_df['layout_id'].isin(hard_ids).values
    p16_hard_mae = abs_residual[hard_mask].mean()
    print(f"Phase 16 hard layouts MAE:           {p16_hard_mae:.4f}")
    print(f"Improvement on hard layouts:         {p13_ranking[p13_ranking['difficulty']=='hard']['mae'].mean() - p16_hard_mae:+.4f}")

layout_stats.to_csv('output/phase16_residual/layout_mae_ranking.csv')

# ============================================================
# Part 4: Scenario 내 timestep 위치별 residual
# ============================================================

print("\n=== Residual by timestep position in scenario ===")

# Scenario 내 row 위치 (0~24)
train_df['row_in_scenario'] = train_df.groupby('scenario_id').cumcount()

position_stats = pd.DataFrame({
    'position': train_df['row_in_scenario'].values,
    'abs_residual': abs_residual,
    'residual': residual,
}).groupby('position').agg(
    mae=('abs_residual', 'mean'),
    residual_mean=('residual', 'mean'),
    count=('abs_residual', 'size')
).round(3)

print(position_stats.to_string())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(position_stats.index, position_stats['mae'], marker='o')
axes[0].set_xlabel('Position in scenario (0-24)')
axes[0].set_ylabel('MAE')
axes[0].set_title('MAE by position in scenario')
axes[0].grid(True, alpha=0.3)

axes[1].plot(position_stats.index, position_stats['residual_mean'], marker='o', color='orange')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_xlabel('Position in scenario')
axes[1].set_ylabel('Residual mean')
axes[1].set_title('Bias by position')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/phase16_residual/residual_by_position.png', dpi=100)
plt.close()
position_stats.to_csv('output/phase16_residual/residual_by_position.csv')

# 첫 row들이 더 어려운가? (lag features가 NaN인 구간)
print(f"\nFirst 3 positions MAE: {position_stats.iloc[:3]['mae'].mean():.4f}")
print(f"Middle 19 positions MAE: {position_stats.iloc[3:22]['mae'].mean():.4f}")
print(f"Last 3 positions MAE: {position_stats.iloc[22:]['mae'].mean():.4f}")

# ============================================================
# Part 5: 피처 값 별 residual (어떤 값 구간에서 틀리나)
# ============================================================

print("\n=== Residual by key feature values ===")

KEY_FEATURES = [
    'pack_utilization', 'congestion_score', 'robot_active',
    'order_inflow_15m', 'charge_queue_length', 'outbound_truck_wait_min',
    'urgent_order_ratio', 'max_zone_density',
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for idx, col in enumerate(KEY_FEATURES):
    if col not in train_df.columns:
        continue

    # Feature value quantile bins
    try:
        q_bins = pd.qcut(train_df[col].fillna(0), q=10, labels=False, duplicates='drop')
    except:
        continue

    stats = pd.DataFrame({
        'bin': q_bins,
        'mae': abs_residual,
        'residual': residual,
    }).groupby('bin').agg(
        mae=('mae', 'mean'),
        residual=('residual', 'mean'),
    )

    ax = axes[idx // 4, idx % 4]
    ax.bar(stats.index, stats['mae'])
    ax.set_title(f'{col}')
    ax.set_xlabel('Quantile bin')
    ax.set_ylabel('MAE')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/phase16_residual/residual_by_feature_value.png', dpi=100)
plt.close()

# 가장 극단 구간 (q=0, q=9)에서 MAE가 큰 피처 찾기
print("\nExtreme quantile residual check:")
for col in KEY_FEATURES:
    if col not in train_df.columns:
        continue
    try:
        q_bins = pd.qcut(train_df[col].fillna(0), q=10, labels=False, duplicates='drop')
    except:
        continue

    df_tmp = pd.DataFrame({'bin': q_bins, 'mae': abs_residual})
    q0_mae = df_tmp[df_tmp['bin'] == 0]['mae'].mean()
    q9_mae = df_tmp[df_tmp['bin'] == 9]['mae'].mean()
    overall_mae = df_tmp['mae'].mean()

    marker = ""
    if q9_mae > overall_mae * 1.3:
        marker = " ★ (high extreme hard)"
    if q0_mae > overall_mae * 1.3:
        marker += " ★ (low extreme hard)"

    print(f"  {col:<30} q0={q0_mae:.2f}, overall={overall_mae:.2f}, q9={q9_mae:.2f}{marker}")

# ============================================================
# Part 6: Top 10 worst predictions (어떤 row가 극단적으로 틀리나)
# ============================================================

print("\n=== Top 10 worst predictions ===")

worst_idx = np.argsort(abs_residual)[::-1][:20]

worst_df = train_df.iloc[worst_idx].copy()
worst_df['y_true'] = y_true[worst_idx]
worst_df['oof_pred'] = oof_p16[worst_idx]
worst_df['residual'] = residual[worst_idx]
worst_df['abs_residual'] = abs_residual[worst_idx]

# 주요 컬럼만 출력
display_cols = ['scenario_id', 'layout_id', 'shift_hour',
                'pack_utilization', 'congestion_score', 'robot_active',
                'order_inflow_15m', 'charge_queue_length',
                'y_true', 'oof_pred', 'residual', 'abs_residual']
display_cols = [c for c in display_cols if c in worst_df.columns]

print(worst_df[display_cols].head(20).to_string())
worst_df.to_csv('output/phase16_residual/worst_predictions.csv', index=False)

# Pattern 확인: 과소예측 vs 과대예측 비율
n_under = (residual > 0).sum()
n_over = (residual < 0).sum()
print(f"\nUnderprediction (residual > 0): {n_under} ({n_under/len(residual)*100:.1f}%)")
print(f"Overprediction (residual < 0): {n_over} ({n_over/len(residual)*100:.1f}%)")

# Top 20 worst 중 과소/과대
top20_under = (worst_df['residual'] > 0).sum()
top20_over = (worst_df['residual'] < 0).sum()
print(f"Top 20 worst: {top20_under} under / {top20_over} over")

# ============================================================
# Part 7: 변동성 높은 scenario에서 residual
# ============================================================

print("\n=== Residual by scenario variability ===")

# Scenario 내 target std
scenario_target_std = train_df.groupby('scenario_id')['avg_delay_minutes_next_30m'].transform('std')
scenario_target_mean = train_df.groupby('scenario_id')['avg_delay_minutes_next_30m'].transform('mean')

# Coefficient of variation
scenario_cv = scenario_target_std / (scenario_target_mean + 1)

# 3개 그룹으로
cv_bins = pd.qcut(scenario_cv, q=3, labels=['low_var', 'mid_var', 'high_var'])

cv_stats = pd.DataFrame({
    'cv_group': cv_bins,
    'abs_residual': abs_residual,
}).groupby('cv_group').agg(
    mae=('abs_residual', 'mean'),
    count=('abs_residual', 'size'),
)

print(cv_stats)
print(f"\nConclusion: {'High variance scenarios are harder' if cv_stats.loc['high_var', 'mae'] > cv_stats.loc['low_var', 'mae'] * 1.2 else 'Variance does not strongly correlate with difficulty'}")

# ============================================================
# Part 8: 결론 및 Phase 17 방향 제안
# ============================================================

print("\n" + "=" * 70)
print("=== PHASE 17 DIRECTION (based on residual analysis) ===")
print("=" * 70)

# 자동 진단
findings = []

# 1. Bin 9 check
bin9_mae = bin_summary.iloc[-1]['abs_mae']
overall_mae = bin_summary['abs_mae'].mean()
if bin9_mae > overall_mae * 3:
    findings.append(f"1. BIN 9 STILL CRITICAL: {bin9_mae:.1f} vs overall {overall_mae:.1f}")
    findings.append("   -> regime detection, extreme value features needed")
elif bin9_mae > overall_mae * 2:
    findings.append(f"1. Bin 9 moderately hard: {bin9_mae:.1f} vs {overall_mae:.1f}")
else:
    findings.append(f"1. Bin 9 not critical: {bin9_mae:.1f}")

# 2. Hard layouts check
top5_mae = layout_stats.head(5)['mae'].mean()
bottom5_mae = layout_stats.tail(5)['mae'].mean()
if top5_mae > bottom5_mae * 2:
    findings.append(f"2. Layout disparity huge: top5 {top5_mae:.2f} vs bottom5 {bottom5_mae:.2f}")
    findings.append("   -> layout-specific features, bottleneck identification needed")
else:
    findings.append(f"2. Layout disparity: top5 {top5_mae:.2f} vs bottom5 {bottom5_mae:.2f}")

# 3. Position effect
first_mae = position_stats.iloc[:3]['mae'].mean()
middle_mae = position_stats.iloc[3:22]['mae'].mean()
if first_mae > middle_mae * 1.2:
    findings.append(f"3. First positions harder: {first_mae:.2f} vs {middle_mae:.2f}")
    findings.append("   -> lag features NaN issue. Improve initial value imputation")
else:
    findings.append(f"3. Position not critical")

# 4. Under vs over
if n_under > n_over * 1.3:
    findings.append(f"4. Systematic underprediction: {n_under} under vs {n_over} over")
    findings.append("   -> Need loss that pushes tail harder (Huber param tuning)")
elif n_over > n_under * 1.3:
    findings.append(f"4. Systematic overprediction")
else:
    findings.append(f"4. Under/over balanced")

for f in findings:
    print(f)

# Save
with open('output/phase16_residual/phase17_direction.md', 'w') as f:
    f.write("# Phase 17 Direction (from Residual Analysis)\n\n")
    for finding in findings:
        f.write(f"- {finding}\n")
    f.write(f"\n## Summary\n")
    f.write(f"- Phase 16 OOF MAE: {overall_mae:.4f}\n")
    f.write(f"- Bin 9 MAE: {bin9_mae:.2f}\n")
    f.write(f"- Hard layout MAE (top 5): {top5_mae:.2f}\n")
    f.write(f"- First 3 positions MAE: {first_mae:.2f}\n")

print("\nResults saved to output/phase16_residual/")
