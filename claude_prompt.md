Phase 21A: fillna(median) 단독 효과 측정

Base: run_phase16_fe.py 복사
Change: NaN cleanup 부분만 fillna(median for 33 cols) + fillna(0 for rest)
        다른 모든 코드는 Phase 16 그대로
        - adversarial weight 없음
        - MLP loss 그대로 (mae)
        - holdout 분석 안 함, CV만 평가

체크포인트: ckpt_phase21a_{model}.pkl
Submission: submission_phase21a.csv

목표: 
- CV가 Phase 16보다 좋으면 → fillna(median)이 효과
- CV가 같거나 나쁘면 → 트리는 NaN native, 효과 없음

작성만, 실행 금지. 커밋: feat: Phase 21A - fillna(median) isolated test