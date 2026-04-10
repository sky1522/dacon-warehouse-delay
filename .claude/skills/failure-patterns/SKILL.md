# Failure Patterns

반복 금지 실패 체크리스트.

## 1. GBDT Extrapolation 한계
- **원인**: 트리 모델은 leaf 평균을 예측 → train max(715)를 넘는 예측 불가
- **증상**: prediction max가 ~120에서 멈춤, Bin 9 MAE 40+
- **금지**: sample weight, quantile loss로 해결하려는 시도 (Phase 13s3)
- **대안**: 피처 엔지니어링으로 극단값 proxy 제공, 앙상블 다양성 확보

## 2. Small-sample MLP 금지
- **기준**: 학습 샘플 < 5,000이면 MLP 사용 금지
- **근거**: Phase 18에서 ~3,786 extreme samples로 MLP → 과적합
- **증상**: train loss 하락, val loss 상승, fold 간 variance 큼
- **대안**: 전체 데이터로 학습하되 sample weight로 극단값 강조

## 3. Layout Target Encoding in GroupKFold(layout_id)
- **원인**: validation fold의 layout_id가 train fold에 없음
- **결과**: 모든 validation row가 global mean으로 매핑 → 피처 무용
- **금지**: layout별 target 통계 (mean, bin9_rate, residual_mean)
- **대안**: layout_info 기반 순수 domain features (pack_ratio, charger_count 등)

## 4. Blend Overfitting Signal
- **경고 조건**: CV 개선 > 0.05인데 Public 개선 < 0.02
- **원인**: OOF matrix에서의 weight overfitting
- **방지**: weight normalization (sum=1), 음수 weight 제한적 허용
- **검증**: hold-out 1 fold로 blend weight 검증

## 5. Raw Features만으로 Sequence Model 금지
- **원인**: Phase 14 GRU가 93 raw features만 사용 (692 engineered 무시)
- **결과**: 트리 + engineered features 대비 큰 열세
- **규칙**: sequence model 사용 시 반드시 engineered features 포함
- **대안**: tree ensemble OOF를 sequence model input으로 활용

## 체크리스트 (새 Phase 시작 전)
- [ ] 극단값 처리가 GBDT extrapolation 한계를 인정하는가?
- [ ] MLP 학습 샘플 > 5,000인가?
- [ ] Target 파생 피처가 CV split과 호환되는가?
- [ ] Cross-phase blend CV 개선이 과적합이 아닌가?
- [ ] Sequence model에 engineered features를 포함하는가?
