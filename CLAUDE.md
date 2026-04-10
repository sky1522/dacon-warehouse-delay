# 데이콘 스마트 창고 출고 지연 예측

## 프로젝트 개요
- **대회**: [데이콘 스마트 물류 창고 출고 지연 예측](https://dacon.io/competitions/official/236437/overview/description)
- **메트릭**: MAE (Mean Absolute Error)
- **마감**: 2026-05-04
- **현재 순위**: 5위 (Public 9.86105)
- **목표**: 1위 (Public 9.78 이하)

## 호칭
- "사장님" 금지. 이름 또는 호칭 없이 대화.

## 워크플로우
- **Claude Code**: 구현 (run_phase*.py 작성)
- **Codex CLI** (`--reasoning high`): 코드 리뷰, 버그 탐지
- **Claude.ai**: 전략 수립, 방향 논의

## 결과 파일 규칙
- `claude_results.md`: Claude Code 작업 결과 (Phase별 섹션)
- `codex_results.md`: Codex 리뷰 결과 (매번 덮어쓰기)
- 미완성 placeholder `?.????` 금지 — 모르면 `[unknown]`으로 표시

## Phase 명명 규칙
- 스크립트: `run_phase{NN}[_suffix].py`
- 체크포인트: `ckpt_phase{NN}[s{seed}]_{model}.pkl`
- 제출: `submission_phase{NN}.csv`

## 데이터 경로
- `data/train.csv`, `data/test.csv`, `data/layout_info.csv`, `data/sample_submission.csv`

## 체크포인트 관리
- 로컬: `output/`
- Drive: `/content/drive/MyDrive/dacon_ckpt/`
- 학습 직후 즉시 Drive 동기화 (마지막 한 번에 몰아서 X)

## 핵심 교훈 (반복 금지)

### Phase 13s3: Tail weight 실패
- 트리 모델은 sample weight로 tail을 더 맞추지 못함
- leaf 평균 예측 구조상 극단값 extrapolation 불가

### Phase 13s5a: Bin9 classifier OOF 실패
- StratifiedGroupKFold(layout_id)에서 layout 기반 target encoding은 validation에서 global mean으로 떨어짐
- 분류기 확률 → 회귀 결합은 CV 악화

### Phase 14: GRU 실패
- 93개 raw features만 사용, 692개 engineered features 미사용
- 데이터 양 대비 sequence model 과적합

### Phase 17: Explosion features 개별 효과 vs 앙상블 효과 불일치
- 개별 모델에서 유용한 피처가 앙상블에서는 MLP에 악영향
- 앙상블 CV 기준으로만 판단해야 함

### Phase 18: Two-stage MLP 실패
- Extreme 샘플 ~3,786개로 MLP 학습 → 과적합
- GBDT normal + MLP extreme 분리가 오히려 CV 악화
