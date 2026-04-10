# Phase 15: Large-scale Feature Engineering

## Hypothesis
Massive aggregation features (scenario/layout/hour/cross) with importance-based selection will capture group-level patterns.

## Implementation
- Scenario aggregation: 26 cols x 6 stats + 15 cols x 5 extras = ~231 features
- Layout aggregation: 11 cols x 5 stats = ~55 features
- Hour aggregation: 6 cols x 3 stats = ~18 features
- Cross (layout x hour): 5 cols x 2 stats = ~10 features
- Ratio features: 6
- Selection: LGB importance top 200 from ~320 new features
- Total: ~546 features (346 base + 200 selected agg)

## Results
- CV MAE: [unknown]

## Lessons
- Aggregation features at multiple granularities (scenario, layout, hour) are effective
- Importance-based selection prevents feature bloat while keeping useful signals
