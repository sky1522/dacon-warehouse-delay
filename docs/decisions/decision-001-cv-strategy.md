# Decision 001: CV Strategy — StratifiedGroupKFold(layout_id)

## 배경
- Phase 1~12: GroupKFold(scenario_id) 사용
- CV-Public 갭: ~1.5 (CV 과낙관)
- Test에 unseen layout이 포함됨 (50/100 layout 미등장)

## 결정
Phase 13s1부터 StratifiedGroupKFold(layout_id, target_bin=5) 전환.

## 근거
- layout_id grouping: test의 unseen layout 상황 시뮬레이션
- target 5-bin stratification: fold 간 target 분포 균형
- scenario grouping은 같은 layout의 다른 scenario를 validation에 넣어 낙관적

## 결과
- Phase 13s1: CV 8.5668, Public 10.0078 (갭 ~1.44)
- Phase 16: CV 8.4403, Public 9.8795 (갭 ~1.44)
- 갭이 안정적 (~1.44)으로 유지됨

## 교훈
- CV 전략 변경은 이전 Phase와의 비교를 어렵게 함
- 한번 정하면 모든 후속 Phase에서 유지해야 함
