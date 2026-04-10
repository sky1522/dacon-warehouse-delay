# Decision 003: Target Transform — raw vs log1p vs sqrt

## 배경
- Target: skew 5.68, max 715.86, median 9.03
- 강한 right-skewed → 변환 없이 학습하면 tail 무시

## 실험 (Phase 7~)
| Transform | 장점 | 단점 |
|-----------|------|------|
| raw + MAE | 해석 용이, 직접 최적화 | tail 무시 |
| log1p + MAE | tail 압축, 분포 정규화 | expm1 역변환 시 오차 증폭 |
| log1p + Huber | robust loss, outlier 완화 | delta 파라미터 민감 |
| sqrt + MAE | 중간 압축 | log1p 대비 덜 효과적 |

## 결정
모든 변환을 동시에 사용 → 앙상블 다양성 확보.

## 근거
- 단일 최적 변환은 없음 (각각 다른 구간에서 강점)
- Nelder-Mead가 변환별 가중치를 자동 결정
- Phase 16 weights: lgb_huber 0.32, mlp 0.45 (log1p 기반이 우세)

## 교훈
- "어떤 변환이 최고인가"보다 "다양한 변환을 앙상블"이 정답
