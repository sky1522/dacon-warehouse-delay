codex --reasoning high "run_phase20_clean.py 리뷰.
컨텍스트:
- Adversarial AUC 0.989 (극심한 shift)
- Phase 16 holdout MAE 9.7341 (Public 9.87에 근접, gap 0.145)
- Top adv features: layout 구조 변수 (aisle_width, compactness, robot_total)

체크:
1. fillna(median) 시점 - Phase 13s1 FE 시작 전인지
2. Adversarial weight 계산 정확성 (proba/(1-proba), clip, normalize)
3. sample_weight 곱셈 순서 (base * adv)
4. Adversarial holdout이 CV folds와 independent한지
5. MLP loss (rmse+mae)/2 tensor 차원
6. Holdout MAE 출력이 최종 결정 지표로 사용되는지
7. 최종 제출은 전체 train으로 재학습하는지 아니면 80% train으로만 학습하는지

결과: codex_results.md"