# 데이콘 스마트 창고 출고 지연 예측 — Phase 3A: 일반화 강화

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 현재 상태
- Phase 2 CV MAE: 8.825, Public MAE: 10.224 (19위)
- CV-Public 갭: 1.40 → 이 갭을 줄이는 것이 최우선 과제
- 앙상블: LightGBM(0.52) + CatBoost(0.33) + XGBoost(0.15)
- 피처: 186개
- test에 train에 없는 layout 50개 포함 (unseen layout 일반화 문제)
- 참고 코드: run_phase2.py

## ⚠️ ID 순서 버그 방지
- concat+sort 후 test 분리 시 반드시 원본 ID 순서 복원
- 제출 파일 생성 시: assert (submission['ID'] == sample_sub['ID']).all()

## ⚠️ 실행 관련
- run_phase3a.py 파일 작성만 할 것. 실행하지 말 것.
- 실행은 내가 직접 PowerShell에서 수행
- 파일 작성 완료 후 "스크립트 작성 완료" 안내

## 수행 작업 (run_phase3a.py)

### 1. 피처 정리 — 중요도 기반 피처 선택
- run_phase2.py의 LightGBM 피처 중요도 기준
- 중요도 하위 30% 피처 제거 → 노이즈 줄이기
- 제거 전후 피처 수 출력
- 제거 후 LightGBM 단독 CV MAE 비교 (제거 전 vs 후)

### 2. 정규화 강화 — 과적합 방지
세 모델 모두 정규화 파라미터 강화:

**LightGBM:**
- reg_alpha=0.1, reg_lambda=1.0, min_child_samples=50
- feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5

**CatBoost:**
- l2_leaf_reg=5.0, min_data_in_leaf=50
- subsample=0.7, colsample_bylevel=0.7

**XGBoost:**
- reg_alpha=0.1, reg_lambda=1.0, min_child_weight=50
- subsample=0.7, colsample_bytree=0.7

### 3. unseen layout 대응 — layout 피처 강화
test에 train에 없는 layout 50개가 있으므로:
- layout_id 자체를 피처로 사용하지 않을 것 (이미 제외되어 있지만 재확인)
- layout 메타데이터 기반 파생 피처 추가:
  * robot_total / pack_station_count (로봇 대비 패킹 스테이션 비율)
  * charger_count / floor_area_sqm (면적당 충전기 밀도)
  * intersection_count / floor_area_sqm (면적당 교차로 밀도)
  * robot_total * layout_compactness (로봇수 × 밀집도 인터랙션)
  * zone_dispersion * robot_total (분산도 × 로봇수)
- layout_type별 통계가 아닌, 수치형 layout 피처 위주로 활용

### 4. 추가 검증 실험 — layout 기반 GroupKFold
- 기존 scenario_id GroupKFold 외에 layout_id GroupKFold도 테스트
- layout_id 기반 CV가 unseen layout 상황을 더 잘 시뮬레이션하는지 확인
- 두 검증 방식의 MAE 비교 출력
- 더 나은 쪽을 최종 검증 전략으로 채택

### 5. 최종 앙상블
- 피처 선택 + 정규화 강화 + layout 피처 추가 적용
- 3모델 앙상블 (가중치 재최적화)
- 최종 CV MAE 출력

### 6. 결과 비교 테이블
=== Phase 3A 결과 ===
실험 1 (피처 선택): LGB CV MAE X.XXXX (186→N개 피처)
실험 2 (정규화 강화): LGB CV MAE X.XXXX
실험 3 (layout 피처 추가): LGB CV MAE X.XXXX
실험 4 (layout GroupKFold): LGB CV MAE X.XXXX
최종 앙상블 CV MAE: X.XXXX
Phase 2 앙상블:     8.8253
개선폭:             X.XXXX

### 7. 제출 파일
- output/submission_phase3a.csv 생성
- ID 순서 assert 검증 필수

## 규칙
- 시각화 한글 폰트 적용
- 모든 MAE print 출력
- 결과를 claude_results.md에 Phase 3A 섹션 추가
- run_phase3a.py 작성만 하고 실행하지 말 것
- 커밋 메시지: "feat: Phase 3A - generalization improvement"
- 커밋 + 푸시