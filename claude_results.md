# Phase 21B: Adversarial Weight 단독 효과 측정

## 목적
- Phase 20에서 도입한 4가지 변경 중 adversarial weight만 분리 테스트
- Phase 16 코드 그대로 + adversarial weight만 추가
- fillna는 Phase 16 그대로 (fillna(0)), MLP loss도 그대로 (mae)

## 변경 사항
- `run_phase16_fe.py` 복사 → `run_phase21b_adv_test.py`
- Adversarial validation: GroupKFold(layout_id), LGBMClassifier
- Weight: proba/(1-proba), clip [0.5, 2.0] (Phase 20의 [0.1, 10]보다 보수적)
- sample_weight = base_weight * adv_weight (8모델 전부 적용, MLP/TabNet 포함)
- 체크포인트: `ckpt_phase21b_{model}.pkl`
- 제출: `submission_phase21b.csv`

## 판단 기준
- CV < 8.4403 → adversarial weight 효과 있음
- CV >= 8.4403 → adversarial weight 무효, Phase 16이 best
