3단계 작업. 각 단계 완료 후 보고. 코드 실행 금지.

## STEP 1: 즉시 정리 (P0)

### 1.1 삭제
- blend_temp.py
- blend_p7p8_60.csv  
- codex_results.old.md
- codex_results.new.md

### 1.2 archive/ 폴더 생성 + 이동
mkdir archive/{phases_01_12,analysis,reviews}

archive/phases_01_12/ 로:
- run_phase1.py ~ run_phase10.py
- run_phase11.py, run_phase11b.py, run_phase12a.py

archive/analysis/ 로:
- run_phase13_step5a.py
- run_phase13s4_bin9_eda.py
- run_phase14_gru.py
- run_phase15b_tabnet.py
- run_phase3b_analysis.py
- run_eda_deep.py

### 1.3 메타데이터 수정
experiments.yaml:
- phase14: submission → submission_phase14_gru.csv
- phase15b: script → run_phase15b_tabnet.py, submission → submission_phase15_full.csv

### 1.4 필수 문서 생성
- README.md (현재 5위 9.86, 빠른 시작, 디렉토리 구조)
- PROGRESS.md (Phase별 한 줄 요약, 현재 → 다음)
- DECISION.md (docs/decisions/ index 역할)
- CHANGELOG.md (구조 변경 이력)

### 1.5 누락 phase 문서
docs/phases/ 추가:
- phase13s2.md (hard layout 분석)
- phase13s4.md (bin9 EDA)
- phase16_residual.md
- phase19.md (multi-seed 실패 회고)

## STEP 2: Phase 20 Pre-EDA 스크립트 작성

run_phase20_eda.py 작성. 4가지 질문:

### Q1: Adversarial AUC
train vs test 분류 가능도 측정
- AUC > 0.7: shift 심각, adversarial weight 효과 클 것
- AUC < 0.55: shift 없음, weight 불필요

### Q2: Median vs Zero Fill 차이
order_inflow_15m, robot_active, battery_mean 3개 column:
- NaN rate
- median 값
- fill(0) vs fill(median) 분포 차이

### Q3: Adversarial Holdout 검증
adv_proba top 20% 추출 후:
- holdout 분포 vs test 분포 vs train_remain 분포 비교
- 5개 핵심 column에서 mean gap 측정

### Q4: Phase 16 OOF의 holdout MAE
phase16 ckpt 복원 후:
- Full CV MAE
- Train remain MAE  
- Holdout MAE ← Public(9.87947)에 가까운지 확인

## STEP 3: Phase 20 본 스크립트 작성 준비

run_phase20_clean.py 작성 (실행 금지):

### 변경 1: fillna(median)
30개 핵심 컬럼 median fill, 나머지는 0

### 변경 2: Adversarial Validation Sample Weight
train_proba / (1 - train_proba), clip [0.1, 10], normalize

### 변경 3: MLP Loss = (RMSE + MAE) / 2

### 변경 4: Adversarial Holdout (top 20% test-like)
학습은 80%, 최종 검증은 holdout MAE

## 규칙
- 작성/이동/삭제만, 코드 실행 절대 금지
- Drive에서 ckpt 복원이 필요하면 그 코드만 작성, 실행은 사용자가
- 각 STEP 완료 후 변경사항 요약
- 커밋 3개:
  1. "chore: P0 cleanup (delete temp, archive old phases, fix experiments.yaml)"
  2. "docs: README/PROGRESS/DECISION/CHANGELOG + missing phase docs"
  3. "feat: Phase 20 pre-EDA + clean preprocessing draft"
- 푸시