# EDA: Train/Test Distribution Shift

## 발견
- 32 features with >5% distribution difference
- 최대 차이: ~53%

## 주요 shift 피처
- 시계열 파생 피처 (lag, rolling)에서 shift 큼
- layout 관련 피처: test에 unseen layout 50개 포함
- 시간대 분포: train과 test의 shift_hour 분포 차이

## Adversarial Validation
- Phase 11에서 시도: AUC ~0.55 (약한 구분력)
- 결론: 분포 차이는 있지만 극단적이지 않음
- Combined weight 적용 → 효과 미미

## 시사점
- Shift가 큰 피처를 무작정 제거하면 오히려 손실
- Adversarial weight는 약한 shift에서 효과 없음
- 더 나은 접근: shift-robust features (rank, percentile 기반)

## Phase 20 계획
- Adversarial validation 재시도 (Phase 17 features 포함)
- Sample weight로 반영 (과도하지 않게 clipping)
