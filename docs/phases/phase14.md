# Phase 14: Bidirectional GRU (FAILED)

## Hypothesis
A sequence model (GRU) operating on 25-timestep scenarios can capture temporal dependencies that tabular models miss.

## Implementation
- Input: per-scenario sequences of 25 timesteps
- Static encoder: layout features → embedding
- Dynamic encoder: bidirectional GRU on ~70 raw dynamic features
- Combined → MLP head for regression
- 3-seed averaging, AdamW + CosineAnnealing
- GroupKFold 5-fold

## Results
- CV MAE: [unknown] (significantly worse than tree ensemble)

## Lessons (CRITICAL)
- **Only 93 raw features used, not 692 engineered features**
  - Tree models benefit from 692+ engineered features (lag, rolling, expanding, interactions, queueing)
  - GRU saw only the raw 93 columns per timestep
  - The feature engineering was the main source of predictive power, not model architecture
- **Data volume insufficient**: ~240K rows / 25 timesteps = ~9,600 scenarios
  - GRU needs much more data to learn temporal patterns from scratch
  - Trees with pre-computed temporal features are more sample-efficient
- **Architecture mismatch**: The problem is better suited for tabular models
  - Each row has rich cross-sectional features (layout, queueing state, etc.)
  - Temporal patterns are already captured by lag/rolling/expanding features
- **If retrying**: Feed all 692 engineered features as sequence input, or use GRU output as additional features for tree ensemble
