# Phase 23 Track A: Codex 리뷰 반영 3개 수정

## Fix 1 (HIGH): layout_type target encoding ValueError
- layout_type = 4 unique → GroupKFold(5) 불가
- KFold(shuffle=True)로 변경

## Fix 2 (MEDIUM): layout_id target encoding 설계 결함 제거
- GroupKFold(groups=layout_id)로 split 시 val layout이 train에 없음
- train OOF가 모두 global_mean (상수) → test와 불일치
- layout_id target encoding 완전 제거 (layout 구조 변수 13개로 충분)

## Fix 3 (LOW): Feature selection 3-fold 평균으로 개선
- 기존: single fold importance
- 수정: 3-fold GroupKFold 평균 importance
- 제거 기준: ALL folds에서 zero OR bottom 5% 평균 importance
