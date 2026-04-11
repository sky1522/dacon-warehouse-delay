# Phase 21A: fillna(median) 단독 효과 측정

## 목적
- Phase 20에서 도입한 4가지 변경 중 fillna(median)만 분리 테스트
- Phase 16 코드 그대로 + NaN cleanup만 median fill로 변경
- adversarial weight 없음, MLP loss 그대로 (mae), holdout 분석 없음

## 변경 사항
- `run_phase16_fe.py` 복사 → `run_phase21a_median_test.py`
- 데이터 로딩 직후 33개 핵심 컬럼에 train median fill 추가
- 나머지 코드 Phase 16과 100% 동일 (8모델 + ensemble)
- 체크포인트: `ckpt_phase21a_{model}.pkl`
- 제출: `submission_phase21a.csv`

## 판단 기준
- CV < 8.4403 → fillna(median) 효과 있음
- CV >= 8.4403 → 트리 모델은 NaN native 처리, 효과 없음
