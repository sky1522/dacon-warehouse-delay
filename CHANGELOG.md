# Changelog

## 2026-04-11

### Structure Cleanup (P0)
- Deleted: blend_temp.py, blend_p7p8_60.csv, codex_results.old.md, codex_results.new.md
- Archived: Phase 1~12 scripts → archive/phases_01_12/
- Archived: Analysis scripts → archive/analysis/
- Fixed: experiments.yaml (phase14 submission, phase15b script/submission)

### Documentation
- Added: README.md, PROGRESS.md, DECISION.md, CHANGELOG.md
- Added: Missing phase docs (phase13s2, phase13s4, phase16_residual, phase19)

### Operational Assets (환경 조성 2차)
- Added: 3 skills (preprocessing, failure-patterns, post-processing)
- Added: 6 decision docs, 5 analysis docs
- Added: 4 scripts (sync_ckpt, blend_cross_phase, check_submission, phase_status)
- Added: 3 templates (phase, decision, phase_doc)
- Added: Phase 20 plan + pre-EDA + clean preprocessing draft

## 2026-04-10

### Environment Setup (환경 조성 1차)
- Added: CLAUDE.md, requirements.txt, .gitignore extension
- Added: 5 skills (phase-runner, checkpoint-manager, phase-results, dacon-cv, blend-optimizer)
- Added: 18 phase retrospective docs
- Added: experiments.yaml ledger
- Added: .claude/settings.local.json hooks

### Phase 18
- Two-Stage Hurdle Model (GBDT normal + MLP extreme + isotonic)

### Phase 17
- Explosion + layout hardness + position + CV² features
- Codex review: layout_residual_bias → hardness indicators, weight normalization
