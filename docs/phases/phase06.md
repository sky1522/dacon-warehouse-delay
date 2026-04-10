# Phase 6: Domain Knowledge Features

## Hypothesis
22 domain-based features (chain bottleneck, robot capacity, fatigue, order-warehouse compatibility) will improve predictions.

## Implementation
- A) Chain bottleneck detection (5): picking_packing_gap, chain_pressure, etc.
- B) Robot available capacity (5): available_capacity, robot_shortage, etc.
- C) Cumulative fatigue (5): scenario_progress, battery_drain_rate, etc.
- D) Order-warehouse compatibility (5): complex_in_narrow, urgent_pack_pressure, etc.
- E) Composite risk (2): risk_score, capacity_stress
- Optuna retune + target transform comparison (log1p/sqrt/raw+MAE/raw+Huber)

## Results
- CV MAE: [unknown]

## Lessons
- Domain features add incremental value but diminishing returns
- Multiple target transforms increase ensemble diversity
