# Phase 20 Bugfix Summary

## 적용된 버그 수정 (4개)

### Fix 1 (Critical): roc_auc_score import 추가
- `from sklearn.metrics import roc_auc_score` 누락으로 스크립트 실행 불가
- `mean_absolute_error` import에 병합

### Fix 2 (Critical): Holdout MAE 비편향 계산
- **기존**: full OOF에서 ensemble weight 최적화 → holdout이 weight 결정에 영향 (biased)
- **수정**: `train_remain_mask = ~holdout_mask`로 80% 데이터에서만 weight 최적화
- holdout 20%는 weight 결정에 미참여 → unbiased Public MAE 근사 가능
- Expected Public range 출력 추가

### Fix 3 (Medium): Adversarial split GroupKFold
- **기존**: StratifiedKFold → 같은 layout_id가 train/valid 양쪽에 존재, classifier가 layout으로 cheat 가능
- **수정**: GroupKFold(groups=layout_id)로 layout 누출 방지
- num_leaves=63, max_depth=7, min_child_samples=100으로 파라미터 조정

### Fix 5 (Critical): MLP/TabNet에 sample_weight 추가
- **MLP**: `mdl.fit(..., sample_weight=sample_w[tri])` 추가
- **TabNet**: `mdl.fit(..., weights=sample_w[tri])` 추가
- adversarial weight가 GBDT에만 적용되고 MLP/TabNet에 누락되었던 문제 해결

## 스킵된 수정

### Fix 4 (Medium): Fold-local median fill → Phase 20.1로 연기
- 구조 변경 크기가 커서 별도 phase에서 처리 예정
- 현재 전역 median fill의 mild leakage 인지만 하고 진행
