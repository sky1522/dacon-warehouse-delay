# Decision 002: Base Model Choice — 7 Models

## 배경
- Phase 2: LGB + Cat + XGB 3모델 시작
- Phase 7~8: target transform 다양화로 6모델
- Phase 10: MLP + TabNet 추가 → 8모델 (TabNet 불안정 → 실질 7모델)

## 최종 구성
1. **LGB raw+MAE**: Optuna params, 가장 안정적
2. **LGB log1p+Huber**: tail 압축 + robust loss
3. **LGB sqrt+MAE**: 중간 압축
4. **XGB raw+MAE**: LGB와 다른 tree 구조
5. **Cat log1p+MAE**: ordered boosting, depth=8
6. **Cat raw+MAE**: depth=6, 다른 정규화
7. **MLP (512-256-128-64)**: log1p target, fold-isolated scaler

## 근거
- 같은 피처에 다른 target transform + loss → prediction 다양성 극대화
- MLP는 GBDT와 구조적으로 다른 예측 → 앙상블 보완
- TabNet: 불안정하지만 성공 시 추가 다양성 (try/except 패턴)

## 교훈
- 7모델이면 Nelder-Mead로 충분한 조합 가능
- 모델 수 > 10은 학습 시간 대비 한계 수익 감소
