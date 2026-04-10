# Phase Results

결과 문서화 및 ledger 관리 패턴.

## claude_results.md 업데이트 규칙
- Phase별 섹션으로 구성
- 실행 후 결과 채우기 (placeholder `?.????` 금지, 모르면 `[unknown]`)
- 필수 항목: CV MAE, Public MAE, 모델별 개별 CV, 앙상블 가중치

## 섹션 템플릿
```markdown
## Phase NN: [제목] (run_phaseNN.py)

### 핵심 전략
- ...

### 결과
- 모델별 CV MAE: ...
- Ensemble CV: X.XXXX
- Public MAE: X.XXXXX (N위)

### 생성된 파일
- `output/submission_phaseNN.csv`
- `output/feature_importance_phaseNN.png`
```

## experiments.yaml ledger
```yaml
- phase: phaseNN
  script: run_phaseNN.py
  date: YYYY-MM-DD
  cv: X.XXXX
  public: X.XXXXX
  rank: N
  submission: submission_phaseNN.csv
  commit: abc1234
  notes: 핵심 변경 요약
```

## 커밋 메시지 규칙
- `feat: Phase NN - [핵심 변경 요약]`
- `fix: Phase NN - [버그 수정 내용]`
- `chore: Phase NN - [문서/환경 변경]`
