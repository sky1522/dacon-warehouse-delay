# 데이콘 스마트 창고 출고 지연 예측 — EDA 시각화 + 베이스라인 모델

## 프로젝트 경로
C:\dev\dacon-warehouse-delay\

## 데이터 경로
data/train.csv, data/test.csv, data/layout_info.csv, data/sample_submission.csv

## 대회 정보
- 타겟: avg_delay_minutes_next_30m (회귀, 강한 우측 꼬리 분포)
- 평가지표: MAE
- train: 250,000행 x 94열 (10,000 시나리오 x 25 타임슬롯)
- test: 50,000행 x 93열 (2,000 시나리오 x 25 타임슬롯)
- layout_info: 300개 레이아웃의 정적 메타데이터 (15열)
- test에 train에 없는 layout 50개 포함 → layout_id 자체를 암기하면 안됨

## Codex 조사에서 확인된 핵심 사항
1. 타겟 분포: mean 18.96, median 9.03, skew 5.68, max 715.86 → log1p 변환 또는 Huber loss 고려
2. 결측치: 86/94 컬럼에 결측 존재, 거의 모든 행에 1개 이상 null, 비율은 11.8~13.4%로 균일 → 행 삭제 불가, 결측 지시자 피처 추가
3. timeslot 컬럼 없음 → groupby("scenario_id").cumcount()로 implicit_timeslot 생성 필수
4. 타겟 상관관계 Top 5: low_battery_ratio(0.37), battery_mean(-0.36), robot_idle(-0.35), order_inflow_15m(0.34), robot_charging(0.32)
5. layout_info 조인키: layout_id, pack_station_count이 타겟과 가장 높은 상관(-0.19)
6. 검증: scenario_id 기준 GroupKFold 사용 (같은 시나리오의 25행이 train/val에 분리되면 안됨)

## 수행 작업

### Phase 1: EDA 시각화 (notebooks/01_eda.ipynb)
1. 타겟 분포: 원본 + log1p 변환 히스토그램 (나란히)
2. 타겟 상관관계 Top 30 수평 막대 그래프
3. implicit_timeslot별 타겟 평균 추이 (라인 차트)
4. layout_type별 타겟 분포 박스플롯
5. 배터리 관련 피처 vs 타겟 산점도 (2x2 서브플롯: low_battery_ratio, battery_mean, robot_charging, charge_queue_length)
6. 결측 비율 Top 20 막대 그래프

### Phase 2: 피처 엔지니어링 + 베이스라인 모델 (notebooks/02_baseline.ipynb)
1. 전처리
   - implicit_timeslot 생성
   - layout_info.csv 조인 (layout_type은 one-hot 또는 label encoding)
   - 정규화 레이아웃 피처: robot_total/floor_area_sqm, charger_count/robot_total, pack_station_count/robot_total
   - 수요 대비 용량 비율: order_inflow_15m/robot_active, unique_sku_15m/pack_station_count
   - 배터리 병목 인터랙션: low_battery_ratio*charge_queue_length, battery_mean-battery_std
   - 결측치: LightGBM은 자체 결측 처리 가능하므로 일단 그대로 두되, 결측 지시자 피처는 상위 10개 컬럼에 대해 추가
   - ID, layout_id, scenario_id는 학습 피처에서 제외
2. 검증 전략
   - scenario_id 기준 GroupKFold (5-Fold)
   - 각 Fold MAE 및 전체 평균 MAE 출력
3. 모델
   - LightGBM Regressor (objective: mae)
   - 하이퍼파라미터: n_estimators=1000, learning_rate=0.05, num_leaves=63, early_stopping_rounds=50
4. 결과
   - CV MAE 점수 출력 (반드시)
   - 피처 중요도 Top 30 시각화
   - test.csv 예측 → output/submission_baseline.csv 생성
   - submission 파일이 sample_submission.csv와 동일 포맷인지 검증

### Phase 3: log1p 타겟 실험 (02_baseline.ipynb 하단에 추가)
1. 동일 파이프라인에서 y = log1p(avg_delay_minutes_next_30m)으로 학습
2. 예측 시 expm1으로 역변환
3. CV MAE 비교 출력: 원본 타겟 vs log1p 타겟
4. 더 좋은 쪽으로 output/submission_best.csv 생성

## 규칙
- 모든 시각화는 한글 폰트 적용 (맑은 고딕 또는 NanumGothic)
- 코드 실행 후 결과 확인까지 완료할 것
- CV MAE 점수를 print로 반드시 출력
- 결과 요약을 claude_results.md에 저장
- Git 설정: 아직 init 안 된 상태. 아래 순서로 진행
  1. git init
  2. .gitignore 생성 (data/*.csv, __pycache__, .ipynb_checkpoints, *.pyc)
  3. git remote add origin https://github.com/sky1522/dacon-warehouse-delay.git
  4. 작업 완료 후 커밋 + 푸시
  5. 커밋 메시지: "feat: EDA visualization and LightGBM baseline"
- data/ 폴더의 csv 파일은 gitignore 처리 (대회 데이터 공유 금지 규정)

## 기술 스택
- Python 3.x, pandas, numpy, matplotlib, seaborn
- scikit-learn (GroupKFold), lightgbm
- jupyter notebook