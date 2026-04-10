# 프로젝트 전수조사 보고서

조사 시점은 2026-04-11 KST 기준이다. 범위는 루트 레벨 `.py/.md/.csv/.json/.yaml`, `run_phase*.py`, `docs/*`, `output/*` 이다. 과거 보고서와 달리 현재 `.gitignore` 는 `output/`, `catboost_info/`, `*.pkl`, `*.png`, `*.csv`, `codex_results*.md` 등을 이미 무시하도록 보강되어 있다.

현재 가장 중요한 결론은 4가지다.

- 루트에는 임시/백업 파일 4개가 남아 있고, `blend_temp.py` 와 `blend_p7p8_60.csv` 는 즉시 정리 후보다.
- `docs/`, `docs/decisions/`, `docs/analysis/`, `.claude/skills/`, `experiments.yaml` 은 이미 존재한다. 예전의 “문서가 비어 있음” 판단은 더 이상 맞지 않는다.
- 반대로 로컬 `output/` 에는 2026-04-11 현재 `ckpt_*.pkl` 이 1개도 없고, 제출물도 `phase10` 이전까지만 남아 있어 `phase13s2` 이후 체인은 재실행이 바로 되지 않는다.
- `experiments.yaml`, `CLAUDE.md`, 실제 스크립트/산출물 이름 사이에 몇 군데 메타데이터 드리프트가 있다.

## 1. 루트 레벨 파일 전수조사

대상 루트 파일은 총 38개다. 루트 `.json` 파일은 없다.

| 파일 | 크기 | 수정 시각 | 추정 목적 | 상태 | 판단 |
| --- | ---: | --- | --- | --- | --- |
| `blend_p7p8_60.csv` | 1,513,427 B | 2026-04-07 01:50:17 | P7/P8 60% 수동 블렌드 산출물 | temporary | 삭제 또는 `output/archive/` 이동 |
| `blend_temp.py` | 747 B | 2026-04-08 01:37:48 | `submission_phase7/8/10` 수동 블렌드 생성 스크립트 | temporary | 삭제 또는 `scripts/adhoc/` 이동 |
| `CLAUDE.md` | 2,196 B | 2026-04-11 03:10:25 | 프로젝트 운영 규칙, 역할 분담, 결과 파일 규칙 | active | 유지 |
| `claude_prompt.md` | 4,692 B | 2026-04-11 03:57:14 | Claude 측 작업 지시 프롬프트, 필요한 skill/checklist 메모 | active | 유지, 주기적 갱신 |
| `claude_results.md` | 2,423 B | 2026-04-11 04:02:39 | 현재는 환경/skill/docs 추가 내역 기록 | active | 유지, 역할 재정의 필요 |
| `codex_prompt.md` | 2,558 B | 2026-04-11 05:06:35 | Codex 측 프로젝트 전수조사 요청 프롬프트 | active | 유지 |
| `codex_results.md` | 신규 작성 | 2026-04-11 | Codex 최신 조사/리뷰 결과의 canonical 파일 | active | 유지, 매 실행 시 덮어쓰기 |
| `codex_results.new.md` | 7,538 B | 2026-04-09 21:47:44 | `run_phase16_fe.py` 과거 리뷰 백업 | duplicate | 삭제 또는 `archive/reviews/` 이동 |
| `codex_results.old.md` | 3,515 B | 2026-04-10 06:15:31 | `run_phase17_fe.py` 과거 리뷰 백업 | duplicate | 삭제 또는 `archive/reviews/` 이동 |
| `experiments.yaml` | 5,324 B | 2026-04-11 03:12:49 | baseline~phase18 실험 ledger | active | 유지, 이름 불일치 수정 필요 |
| `run_eda_deep.py` | 48,148 B | 2026-04-08 22:03:53 | Phase 13 전략용 종합 EDA | unknown | 문서화 후 `scripts/analysis/` 또는 `archive/` 이동 |
| `run_phase1.py` | 7,806 B | 2026-04-03 06:05:51 | Phase 1 시계열 feature baseline | deprecated | 보존 가치 있음, `archive/phases_01_10/` 이동 권장 |
| `run_phase10.py` | 41,366 B | 2026-04-07 22:43:21 | Phase 10 8-model ensemble + ckpt 저장 | deprecated | archive 권장 |
| `run_phase11.py` | 58,060 B | 2026-04-08 01:49:27 | Phase 11 adversarial validation + transformer | deprecated | archive 권장 |
| `run_phase11b.py` | 44,521 B | 2026-04-08 09:11:14 | Phase 11 보정판, transformer 제거 | deprecated | archive 권장 |
| `run_phase12a.py` | 48,561 B | 2026-04-08 17:07:16 | Queueing theory feature 실험 | deprecated | archive 권장 |
| `run_phase13_step1.py` | 55,303 B | 2026-04-08 23:12:42 | StratifiedGroupKFold(layout) 전환 + EDA feature | active | 유지 |
| `run_phase13_step3.py` | 61,682 B | 2026-04-09 02:58:07 | tail-aware modeling + quantile/XGB 변형 | active | 유지 |
| `run_phase13_step5a.py` | 60,227 B | 2026-04-09 11:09:14 | bin9-aware feature 실험 | deprecated | archive 권장 |
| `run_phase13s2_analysis.py` | 20,381 B | 2026-04-09 02:39:08 | hard layout 분석, `layout_mae_ranking.csv` 생성 | active | 유지 |
| `run_phase13s4_bin9_eda.py` | 19,493 B | 2026-04-09 10:20:58 | bin9 극단값 EDA | deprecated | archive 권장 |
| `run_phase14_gru.py` | 19,453 B | 2026-04-09 14:33:50 | scenario-level GRU 실험 | deprecated | archive 권장 |
| `run_phase15_fe.py` | 56,154 B | 2026-04-09 15:45:23 | 대규모 aggregation FE | active | 유지 |
| `run_phase15b_tabnet.py` | 38,960 B | 2026-04-09 20:10:18 | TabNet만 재학습하는 branch | deprecated | rename 또는 archive 필요 |
| `run_phase16_eda.py` | 14,835 B | 2026-04-09 21:07:14 | lag feature 타당성 검증 EDA | active | 유지 |
| `run_phase16_fe.py` | 54,938 B | 2026-04-09 22:01:41 | 2nd-order FE, docs 기준 current best | active | 유지 |
| `run_phase16_residual.py` | 13,910 B | 2026-04-10 05:18:05 | phase16 residual 분석, phase17 방향 설정 | active | 유지 |
| `run_phase17_fe.py` | 71,669 B | 2026-04-10 06:50:03 | explosion/hardness/CV^2 feature 실험 | active | 유지 |
| `run_phase18_twostage.py` | 69,162 B | 2026-04-10 14:11:26 | two-stage hurdle 실험 | active | 유지 |
| `run_phase2.py` | 11,726 B | 2026-04-03 20:24:18 | Phase 2 다중 모델 ensemble baseline | deprecated | archive 권장 |
| `run_phase3a.py` | 19,821 B | 2026-04-03 22:15:47 | feature selection + layout feature | deprecated | archive 권장 |
| `run_phase3b.py` | 22,967 B | 2026-04-04 04:26:03 | residual feature + Optuna tuning | deprecated | 로컬 수정본 정리 후 archive 권장 |
| `run_phase3b_analysis.py` | 22,671 B | 2026-04-04 02:10:21 | phase3b 사전분석/리포트 생성 | unknown | 문서화 후 `scripts/analysis/` 이동 |
| `run_phase5.py` | 22,189 B | 2026-04-05 19:29:21 | 중간 단계 ensemble 개선 | deprecated | archive 권장 |
| `run_phase6.py` | 25,425 B | 2026-04-06 00:00:22 | Optuna 기반 추가 튜닝 | deprecated | archive 권장 |
| `run_phase7.py` | 27,303 B | 2026-04-06 13:55:38 | onset/sample weight 계열 강화 | deprecated | archive 권장 |
| `run_phase8.py` | 38,787 B | 2026-04-06 18:29:27 | stacking/level-1 ensemble | deprecated | archive 권장 |
| `run_phase9.py` | 33,828 B | 2026-04-07 01:18:28 | multiseed tree ensemble | deprecated | archive 권장 |

중복 파일 그룹 식별 결과:

| 그룹 | 구성원 | 판단 |
| --- | --- | --- |
| Codex 리뷰 백업 | `codex_results.md`, `codex_results.old.md`, `codex_results.new.md` | canonical 1개만 유지하고 나머지는 삭제 또는 `archive/reviews/` 이동 |
| 수동 블렌드 워크플로우 | `blend_temp.py`, `blend_p7p8_60.csv`, `output/blend_p10_p7p8_50/60/70.csv` | 재현성이 필요하면 `scripts/adhoc/` + `output/archive/` 로 정리, 아니면 삭제 |
| 초기 제출물 변형 | `output/submission_baseline.csv`, `output/submission_best.csv`, `output/submission_phase1_fixed.csv`, `output/submission_phase5_blend.csv` | 실험 ledger와 연결되지 않는다면 archive 후보 |

루트/워크트리 메모:

- `git status --short -uall` 기준 수정된 tracked 파일은 `claude_prompt.md`, `codex_prompt.md`, `codex_results.md`, `run_phase3b.py` 다.
- untracked 파일은 `blend_temp.py`, `notebooks/03_timeseries_features.ipynb`, `run_phase1.py`, `run_phase2.py`, `run_phase3b_analysis.py` 다.
- 즉, `run_phase1.py`, `run_phase2.py`, `run_phase3b_analysis.py` 는 문서/ledger에는 등장하지만 아직 Git 기준 canonical tracked 상태가 아니다.

## 2. `run_phase*.py` 의존성 맵

공통 raw 입력은 대부분 `data/train.csv`, `data/test.csv`, `data/layout_info.csv` 이고, 제출 스크립트는 보통 `data/sample_submission.csv` 도 함께 사용한다. 아래 표는 raw 입력 외의 추가 CSV/ckpt 의존성과 실제 출력 중심으로 정리했다.

| 스크립트 | 추가 입력(ckpt/csv) | 주요 출력 | 상위 의존 | 현재 판단 |
| --- | --- | --- | --- | --- |
| `run_phase1.py` | 없음 | `output/submission_phase1.csv` | 독립 | 초기 baseline, archive 후보 |
| `run_phase2.py` | 없음 | `output/submission_phase2.csv` | 독립 | archive 후보 |
| `run_phase3a.py` | 없음 | `output/submission_phase3a.csv` | 독립 | `docs/phases/phase03.md` 에 통합 기록 |
| `run_phase3b.py` | 없음 | `output/submission_phase3b.csv` | 독립 | `docs/phases/phase03.md` 에 통합 기록 |
| `run_phase3b_analysis.py` | 내부 계산만 사용 | `output/phase3b_analysis.json`, `output/phase3b_preanalysis.md` | phase3b 맥락 | downstream 참조 없음, orphan 후보 |
| `run_phase5.py` | 없음 | `output/submission_phase5.csv` | 독립 | archive 후보 |
| `run_phase6.py` | 없음 | `output/submission_phase6.csv` | phase3b 패턴 재사용 | 로컬 output 없음 |
| `run_phase7.py` | 없음 | `output/submission_phase7.csv` | phase3b 패턴 재사용 | 로컬 output 존재 |
| `run_phase8.py` | 없음 | `output/submission_phase8.csv` | phase7 이후 stacking branch | 로컬 output 존재 |
| `run_phase9.py` | 없음 | `output/submission_phase9.csv` | 독립 multiseed branch | 로컬 output 존재 |
| `run_phase10.py` | 없음 | `output/ckpt_{lgb,xgb,cat,mlp,tabnet}*.pkl`, `output/submission_phase10.csv` | 독립 | 로컬에는 submission만 있고 ckpt는 전부 없음 |
| `run_phase11.py` | 없음 | `output/ckpt_phase11_*.pkl`, `output/submission_phase11.csv` | phase10 후속 | 로컬 산출물 전부 없음 |
| `run_phase11b.py` | 없음 | `output/ckpt_phase11b_*.pkl`, `output/submission_phase11b.csv` | phase11 successor | 로컬 산출물 전부 없음 |
| `run_phase12a.py` | 없음 | `output/ckpt_phase12a_*.pkl`, `output/submission_phase12a.csv` | phase10 successor | 로컬 산출물 전부 없음 |
| `run_phase13_step1.py` | 없음 | `output/ckpt_phase13s1_*.pkl`, `output/submission_phase13s1.csv` | 새 CV 체계 시작점 | 로컬 산출물 전부 없음 |
| `run_phase13s2_analysis.py` | `output/ckpt_phase13s1_lgb_huber.pkl` | `output/phase13s2_analysis/layout_mae_ranking.csv`, `hard_vs_easy_ks.csv` | phase13s1 | 선행 ckpt 부재로 로컬 실행 불가 |
| `run_phase13_step3.py` | `output/phase13s2_analysis/layout_mae_ranking.csv` | `output/ckpt_phase13s3_*.pkl`, `output/submission_phase13s3.csv` | phase13s2 | 선행 분석 CSV 부재로 로컬 실행 불가 |
| `run_phase13s4_bin9_eda.py` | `output/ckpt_phase13s1_lgb_huber.pkl` | `output/phase13s4_bin9/*.csv` | phase13s1 | downstream 참조 없음, analysis orphan 후보 |
| `run_phase13_step5a.py` | `output/phase13s2_analysis/layout_mae_ranking.csv` | `output/ckpt_phase13s5a_*.pkl`, `output/submission_phase13s5a.csv` | phase13s2 | failed branch, 로컬 실행 불가 |
| `run_phase14_gru.py` | `output/submission_phase13s1.csv` | `output/phase14_gru_oof.csv`, `output/submission_phase14_gru.csv` | phase13s1 | failed branch, 로컬 실행 불가 |
| `run_phase15_fe.py` | `output/phase13s2_analysis/layout_mae_ranking.csv` | `output/ckpt_phase15_*.pkl`, `output/phase15_feature_importance.csv`, `output/submission_phase15.csv` | phase13s2 | 로컬 실행 불가 |
| `run_phase15b_tabnet.py` | `output/ckpt_phase15_*.pkl`, `output/ckpt_phase13s1_{name}.pkl` | `output/ckpt_phase15_tabnet.pkl`, `output/submission_phase15_full.csv`, `output/submission_phase15b_blend.csv` | phase15 + phase13s1 | 이름/ledger 불일치, 로컬 실행 불가 |
| `run_phase16_eda.py` | 없음 | `output/phase16_eda/lag_corr.csv` | 독립 analysis | downstream 코드 참조 없음 |
| `run_phase16_fe.py` | `output/phase13s2_analysis/layout_mae_ranking.csv` | `output/ckpt_phase16_*.pkl`, `output/phase16_feature_importance.csv`, `output/submission_phase16.csv` | phase13s2 | docs 기준 current best, 로컬 실행 불가 |
| `run_phase16_residual.py` | `output/ckpt_phase16_{name}.pkl`, `output/phase13s2_analysis/layout_mae_ranking.csv` | `output/phase16_residual/*.csv` | phase16 + phase13s2 | 분석 가치는 높지만 현재 로컬 산출물 없음 |
| `run_phase17_fe.py` | 없음 | `output/ckpt_phase17_*.pkl`, `output/phase17_feature_importance.csv`, `output/submission_phase17.csv` | phase16 인사이트 기반 | 코드상 직접 file dependency는 없지만 로컬 산출물 없음 |
| `run_phase18_twostage.py` | `output/ckpt_phase16_{name}.pkl` | `output/ckpt_phase18_extreme_mlp.pkl`, `output/phase18_feature_importance.csv`, `output/submission_phase18.csv` | phase16 | phase16 ckpt 부재로 로컬 실행 불가 |

의존성 체인 핵심:

- `phase13s2_analysis` 는 `phase13s1` ckpt 없이는 재생성할 수 없다.
- `phase13_step3`, `phase13_step5a`, `phase15_fe`, `phase16_fe`, `phase16_residual` 은 모두 `output/phase13s2_analysis/layout_mae_ranking.csv` 에 묶여 있다.
- `phase14_gru` 는 `output/submission_phase13s1.csv` 가 필요하다.
- `phase15b_tabnet` 는 `phase15` ckpt와 `phase13s1` ckpt를 동시에 요구한다.
- `phase18_twostage` 는 `phase16` ckpt를 직접 요구한다.

고아 파일 또는 약한 연결 파일:

| 파일 | 이유 | 판단 |
| --- | --- | --- |
| `run_phase3b_analysis.py` | 생성물 `phase3b_analysis.json`, `phase3b_preanalysis.md` 를 다른 스크립트가 읽지 않음 | archive 또는 `scripts/analysis/` 이동 |
| `run_phase13s4_bin9_eda.py` | phase17 방향 설정에는 기여했지만 코드상 downstream 입력으로 쓰이지 않음 | archive 후보 |
| `run_phase16_eda.py` | `lag_corr.csv` 를 직접 읽는 후속 스크립트가 없음 | 유지하되 analysis 영역으로 분리 |
| `run_phase16_residual.py` | phase17 설계 근거 문서로는 유효하지만 file dependency chain 에서는 종료점 | 유지하되 analysis 영역으로 분리 |
| `run_eda_deep.py` | root 단독 EDA 스크립트, 실험 체인과 분리 | 문서화 후 이동 |

삭제 후보 요약:

- 즉시 삭제: `blend_temp.py`, `blend_p7p8_60.csv`, `codex_results.old.md`, `codex_results.new.md`
- 삭제보다는 archive: `run_phase1.py`~`run_phase10.py`, `run_phase14_gru.py`, `run_phase15b_tabnet.py`, `run_phase13s4_bin9_eda.py`, `run_phase3b_analysis.py`, `run_eda_deep.py`

## 3. 문서 파일 현황

### 3-1. 요청된 루트/디렉터리 문서 상태

| 항목 | 존재 | 메모 |
| --- | --- | --- |
| `CLAUDE.md` | 예 | 프로젝트 개요, 역할, 결과 파일 규칙 포함 |
| `README.md` | 아니오 | 최상위 진입 문서 부재 |
| `PROGRESS.md` | 아니오 | 진행 현황 index 부재 |
| `DECISION.md` | 아니오 | 대신 `docs/decisions/*.md` 디렉터리 존재 |
| `CHANGELOG.md` | 아니오 | 변경 이력 루트 index 부재 |
| `claude_results.md` | 예 | 현재는 phase log가 아니라 환경/skill/docs 추가 내역 중심 |
| `codex_results.md` | 예 | canonical Codex 결과 파일 |
| `codex_results.old.md` | 예 | 중복/백업 |
| `codex_results.new.md` | 예 | 중복/백업 |
| `claude_prompt.md` | 예 | 여전히 유효한 작업 지시 프롬프트 |
| `codex_prompt.md` | 예 | 현재 조사 요청 프롬프트 |
| `docs/phases/` | 예 | 20개 문서 존재 |
| `docs/decisions/` | 예 | 6개 문서 존재 |
| `docs/analysis/` | 예 | 5개 문서 존재 |
| `.claude/skills/` | 예 | 8개 skill 존재 |

### 3-2. `docs/phases/*.md` 커버리지

현재 `docs/phases/` 에는 다음이 있다.

- `phase01.md`, `phase02.md`, `phase03.md`
- `phase05.md` ~ `phase12.md`
- `phase13s1.md`, `phase13s3.md`, `phase13s5a.md`
- `phase14.md`, `phase15.md`, `phase16.md`, `phase17.md`, `phase18.md`
- `phase20_plan.md`

판단:

- `phase03.md` 는 3A/3B를 함께 다루므로 `run_phase3a.py`, `run_phase3b.py` 에 대한 상위 문서 역할을 한다.
- `phase11.md` 는 `run_phase11.py` 중심이고 `11b` 전용 문서는 없다.
- `phase12.md` 는 `12a` 를 다루므로 실질적으로 `run_phase12a.py` 커버 문서다.
- `phase15.md` 는 `run_phase15_fe.py` 중심이다. `run_phase15b_tabnet.py` 전용 문서는 없다.
- 전용 문서가 비어 있는 영역은 `phase13s2_analysis`, `phase13s4_bin9_eda`, `phase16_residual`, `run_phase3b_analysis`, `run_eda_deep.py` 다.
- `phase04` 와 `phase19` 문서는 없다. 다만 대응 스크립트도 현재 루트에는 없다.

### 3-3. `docs/decisions/*.md` / `docs/analysis/*.md`

현황:

- `docs/decisions/`: 6개. CV 전략, base model choice, target transform, ensemble method, multi-seed, phase20 preprocessing 정리.
- `docs/analysis/`: 5개. target distribution, NaN patterns, train/test shift, layout overlap, scenario structure 정리.

판단:

- 존재 자체는 충분히 긍정적이다.
- 다만 파일 크기가 대체로 0.8~1.0 KB 수준이라 “실행 매뉴얼” 이라기보다는 짧은 의사결정 메모에 가깝다.
- 루트 `DECISION.md` 가 없는 것은 치명적이지 않지만, index 역할 문서는 있으면 탐색성이 좋아진다.

### 3-4. 문서/ledger 불일치

| 항목 | 현재 상태 | 문제 |
| --- | --- | --- |
| `CLAUDE.md` 의 결과 파일 규칙 | `claude_results.md` 를 “Phase별 섹션” 으로 규정 | 실제 `claude_results.md` 는 환경/skill/docs 추가 로그 중심이라 역할 드리프트 발생 |
| `experiments.yaml` phase14 | `submission_phase14.csv` 로 기록 | 실제 스크립트는 `submission_phase14_gru.csv` 저장 |
| `experiments.yaml` phase15b | `script: run_phase15b.py`, `submission: submission_phase15b.csv` | 실제 파일은 `run_phase15b_tabnet.py`, 산출물은 `submission_phase15_full.csv`, `submission_phase15b_blend.csv` |
| `codex_results.old/new.md` | 루트에 백업 상주 | canonical 결과 파일 정책과 충돌 |

## 4. 체크포인트 / 산출물 파일 현황

### 4-1. 실제 `output/` 현황

2026-04-11 현재 `output/` 하위 디렉터리는 없고, 모든 산출물이 flat 하게 루트 `output/` 아래에만 있다.

| 분류 | 개수 | 실제 상태 |
| --- | ---: | --- |
| `ckpt_*.pkl` | 0 | 로컬 체크포인트 전무 |
| `submission_*.csv` | 13 | baseline, best, phase1, phase1_fixed, phase2, phase3a, phase3b, phase5, phase5_blend, phase7, phase8, phase9, phase10 |
| `blend*.csv` | 9 | `blend_p7p8_*`, `blend_p7p8p9.csv`, `blend_p8p9_50.csv`, `blend_p10_p7p8_*` |
| `feature_importance*.png` | 6 | baseline/phase1/2/3a/3b/5 이미지 |
| 기타 PNG | 6 | EDA 이미지 6개 |
| 기타 리포트 | 2 | `phase3b_analysis.json`, `phase3b_preanalysis.md` |

실제 로컬 제출물 목록:

- `submission_baseline.csv`
- `submission_best.csv`
- `submission_phase1.csv`
- `submission_phase1_fixed.csv`
- `submission_phase2.csv`
- `submission_phase3a.csv`
- `submission_phase3b.csv`
- `submission_phase5.csv`
- `submission_phase5_blend.csv`
- `submission_phase7.csv`
- `submission_phase8.csv`
- `submission_phase9.csv`
- `submission_phase10.csv`

즉, `phase6` 제출물도 현재 로컬에는 없고, `phase11` 이후 제출물은 하나도 없다.

### 4-2. 코드가 기대하는 산출물 vs 실제 존재

| 필수 산출물 | 요구 스크립트 | 현재 상태 |
| --- | --- | --- |
| `output/ckpt_phase13s1_lgb_huber.pkl` | `run_phase13s2_analysis.py`, `run_phase13s4_bin9_eda.py` | 없음 |
| `output/phase13s2_analysis/layout_mae_ranking.csv` | `run_phase13_step3.py`, `run_phase13_step5a.py`, `run_phase15_fe.py`, `run_phase16_fe.py`, `run_phase16_residual.py` | 없음 |
| `output/submission_phase13s1.csv` | `run_phase14_gru.py` | 없음 |
| `output/ckpt_phase15_*.pkl` | `run_phase15b_tabnet.py` | 없음 |
| `output/ckpt_phase16_*.pkl` | `run_phase16_residual.py`, `run_phase18_twostage.py` | 없음 |
| `output/phase16_eda/lag_corr.csv` | 문서상만 의미, 직접 downstream 없음 | 없음 |
| `output/phase16_residual/*.csv` | phase17 방향 기록용 | 없음 |

결론:

- 현재 로컬 저장소만으로는 `phase13s2` 이후 주요 branch를 그대로 재현할 수 없다.
- 특히 `phase15`, `phase16`, `phase18` 은 선행 산출물 또는 ckpt 가 없어서 바로 재실행이 막힌다.
- `phase17` 은 코드상 직접 선행 파일을 읽지 않지만, 실제 실험 맥락은 `phase16_residual` 분석 결과에 기대고 있다.

### 4-3. `claude_results.md` 와 실제 산출물 비교

정규식 기준으로 현재 `claude_results.md` 는 `ckpt_*.pkl` 또는 `submission_*.csv` 를 하나도 언급하지 않는다.

- 언급된 체크포인트: 0개
- 언급된 제출물: 0개

판단:

- “기록 누락” 이라기보다 현재 `claude_results.md` 의 역할이 artifact ledger 가 아니다.
- 따라서 실제 산출물 대조 소스로는 `experiments.yaml` 또는 별도 `artifacts.md` 가 더 적합하다.

### 4-4. 임시 / 중복 산출물

즉시 정리 후보:

- 루트 `blend_p7p8_60.csv`
- 루트 `blend_temp.py`
- `output/submission_phase1_fixed.csv`
- `output/submission_phase5_blend.csv`
- `output/blend_*.csv` 전반

삭제 전 확인이 필요한 파일:

- `output/submission_best.csv`
  - baseline alias 인지, 실제 제출본인지 확인 필요
- `output/phase3b_analysis.json`, `output/phase3b_preanalysis.md`
  - `run_phase3b_analysis.py` 를 계속 유지할지 먼저 결정해야 함

## 5. 정리 권장 사항

### A. 즉시 삭제 가능

- `blend_temp.py`
- `blend_p7p8_60.csv`
- `codex_results.old.md`
- `codex_results.new.md`

조건부 즉시 삭제:

- `output/submission_phase1_fixed.csv`
- `output/submission_phase5_blend.csv`
- `output/blend_*.csv`

조건:

- `experiments.yaml` 또는 별도 ledger 에서 해당 파일을 canonical 제출본으로 쓰지 않는다고 확정할 때

### B. archive 권장

- `run_phase1.py` ~ `run_phase10.py`
- `run_phase11.py`, `run_phase11b.py`, `run_phase12a.py`
- `run_phase13_step5a.py`
- `run_phase13s4_bin9_eda.py`
- `run_phase14_gru.py`
- `run_phase15b_tabnet.py`
- `run_phase3b_analysis.py`
- `run_eda_deep.py`

권장 구조:

- `archive/phases_01_12/`
- `archive/analysis/`
- `archive/reviews/`

### C. 리팩터 필요

- `phase15/16/17/18` 공통 feature engineering, CV, ckpt 저장/로딩, ensemble 로직을 모듈로 분리
- artifact path 규칙을 한 곳에서 관리하는 `artifacts.py` 또는 `config.py` 추가
- `experiments.yaml` 과 실제 파일명을 자동 검증하는 스크립트 추가
- analysis 스크립트를 `run_phase*.py` 와 분리해 `scripts/analysis/` 아래로 이동

### D. 문서화 필요

- `README.md`
  - 현재 best phase
  - 재현 순서
  - 필수 의존 파일
- `PROGRESS.md`
  - baseline~phase18 결과 one-line 요약
- `DECISION.md`
  - 이미 있는 `docs/decisions/*.md` 로 가는 index 역할만 해도 충분
- `CHANGELOG.md`
  - 최근 구조 변경과 archive 이력 기록
- 누락된 phase/analysis 문서
  - `phase13s2`
  - `phase13s4`
  - `phase15b`
  - `phase16_residual`
  - `phase3b_analysis`

### E. 업데이트 필요

- `experiments.yaml`
  - phase14 제출물명 수정
  - phase15b script/output 이름 수정
- `CLAUDE.md`
  - `claude_results.md` 역할을 현실에 맞게 수정하거나, 실제 phase log 파일을 따로 분리
- Git 정리
  - `run_phase1.py`, `run_phase2.py`, `run_phase3b_analysis.py` 를 commit 할지 archive 할지 결정
- artifact manifest 추가
  - “현재 로컬에는 phase10까지만 실물 있음” 을 명시한 문서 필요

## 6. 우선순위 정리 계획

### P0 (오늘)

- `experiments.yaml` 의 phase14/phase15b 이름 불일치 수정
- `blend_temp.py`, `blend_p7p8_60.csv`, `codex_results.old.md`, `codex_results.new.md` 정리
- `README.md` 와 `PROGRESS.md` 최소본 생성
- `CLAUDE.md` 와 `claude_results.md` 역할 충돌 정리

### P1 (1~2일)

- `phase13s1` / `phase13s2` / `phase16` 필수 산출물의 실제 보관 위치 확인
- 로컬에 없다면 재생성 순서 문서화
- 초기 phase 스크립트와 output 산출물을 `archive/` 로 이동하는 기준 확정

### P2 (이번 주)

- `phase15~18` 공통 코드 리팩터
- analysis 스크립트 분리
- artifact validation 스크립트 추가

### P3 (다음 주)

- `docs/phases` 누락 영역 보강
- `docs/decisions` / `docs/analysis` 를 README index 와 연결
- 실험 재현 체크리스트 자동화

최종 판단:

- 현재 저장소는 “문서가 없다” 상태는 아니고, 오히려 문서/skills/ledger 는 많이 보강됐다.
- 문제의 중심은 문서 부재보다 artifact 부재와 메타데이터 불일치다.
- 정리 우선순위는 `루트 임시 파일 삭제` 보다 `실제 재현 가능 체인 복원` 이 더 높다.
