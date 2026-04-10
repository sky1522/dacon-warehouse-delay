# EDA: Layout Overlap (Train vs Test)

## 기본 통계
- Train layouts: ~100개 (layout_id 기준)
- Test layouts: ~100개
- **Overlap: ~50개 (50%)**
- Unseen in test: ~50개

## 시사점
- Test의 절반이 train에서 본 적 없는 layout
- → GroupKFold(layout_id) CV가 적절 (unseen layout 시뮬레이션)
- → layout-specific features (target encoding)는 50% test에서 무효

## Layout 난이도
- Hard top 5: WH_051, WH_073, WH_217, WH_049, WH_098
  - Phase 16 hard layout MAE: 30.80 (전체 8.44 대비 3.6배)
  - 공통점: pack_station_count 적음, robot_total 대비 capacity 부족

## CV 전략 영향
- StratifiedGroupKFold(layout_id): validation에 unseen layout 보장
- 이전 GroupKFold(scenario_id): 같은 layout의 다른 scenario가 train/val에 분산 → 낙관적
