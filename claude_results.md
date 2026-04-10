# 환경 조성 2차 — 운영 자산 추가

## 1. Skills 추가 (3개)
- `.claude/skills/preprocessing/SKILL.md` — NaN 전략 (fillna(0) 금지, median), IQR outlier, train/test shift, adversarial validation
- `.claude/skills/failure-patterns/SKILL.md` — GBDT extrapolation, small-sample MLP 금지, layout target encoding leakage, blend overfit, raw-only sequence model 금지
- `.claude/skills/post-processing/SKILL.md` — isotonic nested CV, clip 전략, rank averaging vs value averaging

## 2. docs/decisions/ (6개)
- `decision-001-cv-strategy.md` — StratifiedGroupKFold(layout_id) 선택 이유, 갭 ~1.44 안정
- `decision-002-base-model-choice.md` — 7 models (LGB x3 + XGB + Cat x2 + MLP) 구성 근거
- `decision-003-target-transform.md` — raw/log1p/sqrt 동시 사용 → 앙상블 다양성
- `decision-004-ensemble-method.md` — Nelder-Mead 선택 (stacking 대비 우위), normalization 필수
- `decision-005-multi-seed.md` — 42/2024/777 3-seed averaging
- `decision-006-phase20-preprocessing.md` — fillna(median) + adversarial weight + MLP loss 변경 계획

## 3. docs/analysis/ (5개)
- `eda-target-distribution.md` — skew 5.68, Bin 9 = 48% of MAE, log1p 변환 효과
- `eda-nan-patterns.md` — 86 cols x 12% NaN, MCAR, fillna(0) 왜곡 문제
- `eda-train-test-shift.md` — 32 features >5% diff, 최대 53%, adversarial AUC ~0.55
- `eda-layout-overlap.md` — 50% overlap, unseen layout 50개, hard top 5 MAE 30.80
- `eda-scenario-structure.md` — 25 timesteps, position별 MAE 패턴, CV² 유용성

## 4. scripts/ (4개)
- `scripts/sync_ckpt.py` — Drive <-> local 양방향 동기화 (push/pull/status)
- `scripts/blend_cross_phase.py` — 전체 Phase OOF 로드 → Nelder-Mead 30-model blend
- `scripts/check_submission.py` — ID 정렬, NaN, range, quantile 검증
- `scripts/phase_status.py` — experiments.yaml 파싱 → 현황 테이블 출력

## 5. templates/ (3개)
- `templates/phase_template.py` — 새 Phase base (save_ckpt/load_ckpt + TODO sections)
- `templates/decision_template.md` — 의사결정 문서 양식
- `templates/phase_doc_template.md` — Phase 회고 양식

## 6. docs/phases/phase20_plan.md
- 가설: fillna(0) 왜곡 + shift 무시 + MLP MAE loss → 깨끗한 전처리로 -0.05~0.08
- 5 steps: fillna(median), adversarial weight, MLP loss, multi-seed, Optuna
- 예상: 9.86 → 9.77~9.83
- 의존성: Seed 777 완료, GPU 6시간
