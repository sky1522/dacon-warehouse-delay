Phase 21B: Adversarial Weight 단독 효과 측정

Base: run_phase16_fe.py 복사 → run_phase21b_adv_test.py
Change: 
1. fillna는 Phase 16 그대로 (fillna(0))
2. Adversarial weight 추가 (Phase 20과 동일하지만 약화):
   - GroupKFold(layout_id) classifier
   - clip [0.5, 2.0] (Phase 20은 [0.1, 10] 너무 극단)
   - sample_weight = base_weight * adv_weight
   - MLP/TabNet에도 적용
3. 다른 변경 없음 (MLP loss 그대로 mae)

체크포인트: ckpt_phase21b_*.pkl
Submission: submission_phase21b.csv

목표:
- Ensemble CV < 8.4403 → adversarial weight 효과
- 같거나 나쁨 → adversarial weight도 효과 없음 (Phase 16이 진짜 best)

작성만, 실행 금지. 커밋: feat: Phase 21B - adversarial weight isolated test