# Dacon Smart Warehouse Delay Prediction

**현재 5위** — Public MAE 9.86105 (목표: 1위 9.78)

데이콘 스마트 물류 창고 출고 지연 예측 대회. 마감 2026-05-04.

## Quick Start

```bash
# Colab에서 실행
!pip install -q lightgbm xgboost catboost pytorch-tabnet
!python run_phase16_fe.py  # Current best (CV 8.4403, Public 9.8795)
```

## Directory Structure

```
├── run_phase*.py          # Active phase scripts (13s1~18)
├── archive/
│   ├── phases_01_12/      # Phase 1~12 scripts
│   └── analysis/          # EDA/analysis scripts
├── data/                  # train.csv, test.csv, layout_info.csv
├── output/                # checkpoints, submissions, plots
├── docs/
│   ├── phases/            # Phase retrospectives
│   ├── decisions/         # Architecture decisions
│   └── analysis/          # EDA documentation
├── scripts/               # Utility scripts
├── templates/             # Templates for new phases
├── .claude/skills/        # Claude Code skills
├── CLAUDE.md              # Project guide for Claude
├── PROGRESS.md            # Phase progress tracker
├── DECISION.md            # Decision index
├── CHANGELOG.md           # Structure changes
├── experiments.yaml       # Results ledger
└── requirements.txt       # Python dependencies
```

## Key Results

| Phase | CV | Public | Rank |
|-------|-----|--------|------|
| Phase 16 | 8.4403 | 9.8795 | 5 |
| Phase 13s1 | 8.5668 | 10.008 | - |
| Phase 1 | 8.8508 | 10.249 | 14 |

## Workflow

Claude Code (구현) → Codex CLI (리뷰) → Claude.ai (전략)
