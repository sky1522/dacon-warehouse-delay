# 3단계 정리 + Phase 20 준비

## STEP 1: P0 Cleanup (30d29c0)
- **삭제**: blend_temp.py, blend_p7p8_60.csv, codex_results.old.md, codex_results.new.md
- **archive/phases_01_12/**: run_phase1~12 (12개 스크립트)
- **archive/analysis/**: run_phase13_step5a, 13s4_bin9_eda, 14_gru, 15b_tabnet, 3b_analysis, eda_deep (6개)
- **experiments.yaml 수정**: phase14 submission → submission_phase14_gru.csv, phase15b script → run_phase15b_tabnet.py + submission → submission_phase15_full.csv

## STEP 2: 문서 (25dac9f)
- **README.md**: 현재 5위 9.86, Quick Start, 디렉토리 구조, Key Results
- **PROGRESS.md**: Phase별 한 줄 요약 테이블, Next Steps
- **DECISION.md**: docs/decisions/ 6개 의사결정 index
- **CHANGELOG.md**: 2026-04-10~11 구조 변경 이력
- **누락 phase docs 4개**: phase13s2 (hard layout), phase13s4 (bin9 EDA), phase16_residual, phase19 (multi-seed blend)

## STEP 3: Phase 20 스크립트 (682e2ea)

### run_phase20_eda.py (Pre-EDA, 4가지 질문)
- Q1: Adversarial AUC (5-fold OOF LGBMClassifier, threshold 판단)
- Q2: Median vs Zero fill 차이 (3개 핵심 컬럼 NaN rate, median, 분포 비교)
- Q3: Adversarial holdout (top 20% test-like) 분포 검증
- Q4: Phase 16 OOF holdout MAE (Public 9.87947 근접 여부)

### run_phase20_clean.py (본 스크립트, 4가지 변경)
1. **fillna(median)**: 30개 핵심 컬럼 train 기준 median fill (fillna(0) 대체)
2. **Adversarial weight**: train_proba/(1-train_proba), clip [0.1,10], normalize → sample_weight에 곱셈
3. **MLP loss**: MAE → (RMSE+MAE)/2 (rmse_mae_loss custom function)
4. **Adversarial holdout**: top 20% test-like train samples로 별도 MAE 검증

- 8 models (Phase 17 FE base + 4 changes)
- Checkpoint: ckpt_phase20_{model}.pkl
- Submission: submission_phase20.csv
