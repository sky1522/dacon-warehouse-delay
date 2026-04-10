# Dacon CV Strategy

CV (Cross-Validation) 전략 및 leakage 방지 패턴.

## 표준 CV: StratifiedGroupKFold
```python
from sklearn.model_selection import StratifiedGroupKFold

y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(sgkf.split(X, y_binned, groups=layout_ids))
```

### 핵심 원칙
- **Group**: `layout_id` — unseen layout 검증 (Public 상황 재현)
- **Stratify**: target 5분위 binning — fold 간 target 분포 균형
- **folds 객체 공유**: 분류기/회귀기/calibration 모두 같은 fold split 사용

## Leakage 방지 체크리스트
1. Target encoding (layout별 통계) → OOF 방식 필수
2. StandardScaler → fold-internal fit/transform (MLP)
3. Feature selection → GroupKFold(같은 split) 사용
4. Isotonic calibration → nested OOF (같은 folds)
5. layout_id 기반 GroupKFold에서 layout target encoding → global mean 문제

## Adversarial Validation (실패 경험)
- Phase 11에서 시도, 효과 없었음
- train/test 분포 차이가 크지 않은 대회에서는 불필요

## CV-Public 갭 관리
- 갭 > 1.5: 과적합 의심 → 정규화 강화
- 갭 < 0.5: 안정 → 공격적 피처 추가 가능
- 현재 Phase 16: CV 8.4403, Public 9.87947 (갭 ~1.44)
