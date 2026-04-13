# Phase 18 Recovery: H1 (log1p target) + H3 (composite features)

## 설계 원칙
- Phase 16 구조 100% 유지 (692 features, 8 models, StratifiedGroupKFold)
- 변경 단 2가지만

## 변경 1: H1 - Target Transform
- ALL 8 models train on log1p(y), predict with expm1()
- Model 2 (LGB Huber), Model 5 (Cat log1p), Model 7 (MLP): 이미 log1p → 변경 없음
- Model 1 (LGB MAE), 3 (LGB sqrt→log1p), 4 (XGB), 6 (Cat raw→log1p), 8 (TabNet): log1p로 변경

## 변경 2: H3 - Composite Features 3개
- `service_impediment`: (congestion + charge_queue + blocked + collision) / 4 정규화
- `crisis_indicator`: pack_utilization * service_impediment
- `paradox_indicator`: pack_utilization * (1 - service_impediment)
- Total features: 692 + 3 = 695

## 검증 기준
- Phase 16 재현: CV 8.35~8.45
- Pass: Public <= 9.86
- Success: Public < 9.80
