# Phase 13 Step 1: CV Redesign + EDA Features

## Hypothesis
Switching from GroupKFold(scenario_id) to StratifiedGroupKFold(layout_id) will better simulate Public leaderboard conditions (unseen layouts).

## Implementation
- CV: StratifiedGroupKFold(layout_id, target_bin=5)
- 7 EDA features: shift regime(3), time step(2), overload detection(2)
- Removed 22 low-correlation layout static features
- Kept 42 queueing theory features

## Results
- CV MAE: 8.5668
- Public MAE: 10.0078

## Lessons
- layout_id grouping is critical for realistic validation
- CV-Public gap of ~1.44 is acceptable
- This became the standard CV strategy for all subsequent phases
