# Decision 005: Multi-Seed — 42 / 2024 / 777

## 배경
- Phase 9: 3-seed blending (42, 123, 777) 시도
- 단일 seed의 variance를 줄여 안정적 예측

## 결정
Seed 42 (기본) + 2024 + 777로 3-seed averaging.

## 근거
- Seed 간 CV 차이: ~0.01~0.03 (무시 못할 수준)
- 3-seed 평균: variance 감소, Public 안정성 향상
- 학습 시간 3배이지만 Colab GPU 기준 수용 가능

## 구현
```python
for seed in [42, 2024, 777]:
    # seed 변경 → 학습 → ckpt 저장
    # ckpt_phase16s42_lgb_raw.pkl, ckpt_phase16s2024_lgb_raw.pkl, ...

# 최종: 3 seed OOF 평균
oof_avg = (oof_s42 + oof_s2024 + oof_s777) / 3
```

## 교훈
- Multi-seed는 "무조건 좋다"이지만 시간 예산과 trade-off
- Cross-phase blend와 결합 시 효과 극대화
