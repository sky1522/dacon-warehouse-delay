# Phase 17: Bin 9 Attack — Explosion + Hardness + CV² (MIXED)

## Hypothesis
Based on Phase 16 residual analysis (Bin 9 MAE = 40.92 = 48% of total):
1. Queueing explosion terms rho/(1-rho) will flag near-saturation states
2. Layout hardness indicators will capture structural difficulty
3. CV² features (M/G/1 formula) will capture service time variability

## Implementation
- A) Bottleneck explosion (12): M/M/1 explosion terms for pack/robot/charge/truck
- B) Demand-supply gap (7): weighted demand vs effective supply
- C) Position features (4): scenario position + interaction with explosion/gap
- D) Layout hardness indicators (6): l_pack_ratio, l_pack_severity, l_robot_density, l_effective_capacity, l_capacity_demand_ratio, l_hardness_score
- E) CV² features (12): 6 cols x (CV + CV²)
- Selection: top 30 from ~41 new features
- Total: ~726 features

### Codex Review Fixes
- Category D redesigned: layout_residual_bias (OOF target encoding, would fail under layout GroupKFold) → pure domain features
- Ensemble weight normalization: internal + final normalize
- Dropped column guard: floor_area_sqm was already removed in Phase 13s1

## Results
- CV MAE: [unknown]

## Lessons
- **Individual feature value != ensemble value**: explosion features helped individual LGB but hurt MLP → net ensemble effect unclear
- Layout hardness indicators are safe (no target leakage) but may overlap with existing layout aggregation features
- Always verify that features used in new code still exist in the dataframe (Phase 13s1 column drops)
