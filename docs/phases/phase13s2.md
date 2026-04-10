# Phase 13 Step 2: Hard Layout Analysis

## Hypothesis
Which layouts are hardest to predict? What makes them hard?

## Implementation
- Layout별 MAE ranking (Phase 13s1 OOF 기반)
- Hard/medium/easy difficulty 분류
- Hard layout 공통점 분석: pack_station_count, charger_count, robot_total 비율

## Results
- Hard top 5: WH_051, WH_073, WH_217, WH_049, WH_098
- Hard layout MAE: ~30 (전체 8.57 대비 3.5배)
- 공통점: pack_station_count 적음, robot 대비 capacity 부족

## Lessons
- Layout 난이도는 구조적 (pack/robot ratio가 지배적)
- Hard layout에서 Bin 9 발생 빈도 높음
- → Phase 17 layout hardness indicators 근거
