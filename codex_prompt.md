# 역할: 데이콘 "스마트 창고 출고 지연 예측 AI 경진대회" EDA 조사

## 배경
- 대회: 데이콘 월간 대회, 스마트 물류창고 출고 지연 예측
- 타겟: avg_delay_minutes_next_30m (향후 30분간 평균 출고 지연 시간, 회귀)
- 평가지표: MAE
- 데이터: train.csv, test.csv, layout_info.csv, sample_submission.csv
- 경로: C:\dev\dacon-warehouse-delay\data\

## 수행 작업
1. 각 파일의 shape, columns, dtypes, head(10) 확인
2. train.csv의 90개 피처 목록과 의미 추정 (컬럼명 기반)
3. 타겟(avg_delay_minutes_next_30m) 분포 분석 — 평균, 중앙값, 왜도, 이상치
4. 결측치/중복값 현황 정리
5. 피처 간 상관관계 Top 20 (타겟과의 상관계수 기준)
6. layout_info.csv와 train.csv의 조인 키 및 활용 방안 분석
7. 시나리오(scenario_id) 및 타임슬롯(timeslot) 구조 파악 — 12,000 시나리오 × 25 타임슬롯인지 확인
8. 피처 카테고리 분류 (주문 관련, 로봇 관련, 배터리 관련, 혼잡도 관련, 패킹 관련 등)

## 출력
- 분석 결과를 codex_results.md에 저장
- 핵심 발견사항을 상단에 요약
- 후속 피처 엔지니어링 및 모델링 방향 제안 포함