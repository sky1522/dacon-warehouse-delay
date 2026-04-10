# Phase 12a: Queueing Theory Features

## Hypothesis
42 features based on queueing theory (Little's Law, Pollaczek-Khinchin formula, bottleneck detection) will capture system dynamics.

## Implementation
- A) Utilization/traffic intensity (16): rho for 4 stations + sq/inv/P-K
- B) Little's Law waiting time (6): arrival rate, expected wait, service gap
- C) Bottleneck detection (8): max/min/mean/std/gap + cascade
- D) Queue stability indicators (4): unstable flags + count
- E) Demand surge/time-varying (8): rho change/accel, queue growth
- Phase 10 structure (8 models) maintained

## Results
- CV MAE: [unknown]

## Lessons
- Queueing theory features provide domain-grounded representations
- Some overlap with existing features (diminishing marginal value)
