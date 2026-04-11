# Phase 21C: MLP RMSE+MAE Loss 단독 효과 측정

## 목적
- Phase 20에서 도입한 MLP loss 변경만 분리 테스트
- Phase 16 코드 그대로 + MLP loss만 mae → (RMSE+MAE)/2로 변경

## 변경 사항
- `run_phase16_fe.py` 복사 → `run_phase21c_mlploss_test.py`
- MLP compile: `loss='mae'` → `loss=rmse_mae_loss`
- 트리 모델 7개, TabNet, CV 전략 등 모두 Phase 16과 100% 동일
- 체크포인트: `ckpt_phase21c_{model}.pkl`
- 제출: `submission_phase21c.csv`

## 판단 기준
- MLP CV < 8.5887 → loss 변경이 MLP 자체 성능 개선
- Ensemble CV < 8.4403 → 앙상블에도 기여
- 트리 모델 CV는 변화 없어야 함 (변경 없으므로)
