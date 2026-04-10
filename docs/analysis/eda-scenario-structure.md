# EDA: Scenario Structure

## 기본 구조
- 1 scenario = 25 timesteps (고정)
- 각 timestep = 1 row
- implicit_timeslot: groupby cumcount로 생성 (0~24)

## 시간 순서
- Phase 16 EDA 검증: scenario 내 시간 순서 확인
- shift_hour는 NOT monotonic (같은 scenario 내에서 동일하거나 변화)
- implicit_timeslot이 실제 시간 순서를 나타냄

## Timestep별 특성
- Position 0~2 (초기): lag features NaN, MAE 상대적으로 높음
- Position 3~21 (중기): MAE 안정적
- Position 22~24 (후기): MAE 상승 (delay 누적 효과)

## Phase 16 Residual 분석
- First 3 positions MAE: 높음 (lag NaN 영향)
- Middle 19 positions MAE: 안정
- Last 3 positions MAE: 높음 (과소예측 경향)

## 시사점
- Position feature (time_step_in_scenario, ts_squared)가 유용
- 후반부 과소예측 → sample weight에 time fraction 반영 (Phase 7~)
- Scenario 내 변동성 (CV)도 유용한 피처 (Phase 17 CV²)
