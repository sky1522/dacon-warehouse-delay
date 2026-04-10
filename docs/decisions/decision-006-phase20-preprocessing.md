# Decision 006: Phase 20 — Clean Preprocessing

## 배경
- 지금까지 모든 Phase가 fillna(0) 기반 → 핵심 피처 왜곡
- Train/Test distribution shift 미처리 (32 features >5% diff, 최대 53%)
- MLP loss가 MAE → tail 영역 소홀

## 결정 (Phase 20 계획)
1. fillna(median) for 핵심 3개 피처 (order_inflow_15m, robot_active, battery_mean)
2. Adversarial validation sample weight
3. MLP loss → (RMSE + MAE) / 2
4. Multi-seed (42, 2024, 777)
5. Optuna HPO for LGB huber + MLP

## 예상 효과
- 보수적: 9.86 → 9.83 (4위)
- 중립: 9.86 → 9.80 (2위)
- 낙관: 9.86 → 9.77 (1위)

## 의존성
- Seed 777 학습 완료 대기
- 전처리 변경 → 체크포인트 전부 재학습 필요
- GPU 예산: ~6시간 (Colab T4)

## 리스크
- 전처리 변경 시 기존 Phase 체크포인트 무효화
- fillna(median) 계산이 train 기준이어야 함 (test leakage 방지)
- Adversarial weight가 과도하면 오히려 악화
