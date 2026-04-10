# EDA: NaN Patterns

## 기본 통계
- 86/94 컬럼에 결측 존재
- 결측 비율: 11.8~13.4%로 균일
- 패턴: MCAR (Missing Completely At Random)

## Position 분석
- NaN은 scenario 내 위치와 무관 (uniform distribution)
- 첫 timestep에 lag/rolling NaN은 자연스러움 (feature engineering 부산물)

## 핵심 피처 결측
- order_inflow_15m: ~12% NaN → fillna(0)은 "주문 0건"으로 왜곡
- robot_active: ~12% NaN → fillna(0)은 "로봇 0대"로 왜곡
- battery_mean: ~12% NaN → fillna(0)은 "배터리 0%"로 왜곡

## 현재 처리 (Phase 1~18)
- 대부분 fillna(0) 사용 → 왜곡 발생
- Missing indicator 10개 생성 (Phase 13s1)

## Phase 20 개선 계획
- 핵심 3개 피처: fillna(median) (train 기준)
- 나머지: 기존 방식 유지 (영향 작음)
