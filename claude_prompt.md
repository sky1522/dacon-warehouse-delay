# Experiment 1: Phase 18-Recovery (H1+H3)

## 배경
- Phase II 분석 완료 (first-principles 기반)
- Phase 23 실패 원인: 외부 사례 모방, 복잡도 과잉, feature noise 과다
- 이번에는 Phase 16의 검증된 구조에 **2가지만** 변경하여 개선

## 핵심 변경사항 (단 2가지)

### 변경 1: Target 변환 (H1)
- Raw target: `avg_delay_minutes_next_30m` (skewness 5.68, kurtosis 64)
- **변경**: `y_transformed = np.log1p(y)` (skewness 0.08, kurtosis -0.36)
- Box-Cox λ=0.01 → log 변환이 수학적 최적
- 예측 시 `np.expm1()` 역변환 후 clip(0, None)

### 변경 2: Composite Features 3개 추가 (H3)
Phase 16 base features 유지하고 **단 3개만** 추가:

```python
# 기존 features 그대로 유지 (Phase 16 기반 ~692개)

# 새로 추가할 3개 composite features:
def add_composite_features(df):
    # Service impediment: 0~1 정규화된 복합 지표
    congestion_norm = (df['congestion_score'] / 100).fillna(0).clip(0, 1)
    charge_queue_norm = (df['charge_queue_length'] / 30).fillna(0).clip(0, 1)
    blocked_norm = (df['blocked_path_15m'] / 14).fillna(0).clip(0, 1)
    collision_norm = (df['near_collision_15m'] / 9).fillna(0).clip(0, 1)
    
    df['service_impediment'] = (
        congestion_norm + charge_queue_norm + blocked_norm + collision_norm
    ) / 4
    
    # Crisis indicator: 포화 + 방해 → 크리시스
    df['crisis_indicator'] = df['pack_utilization'].fillna(0) * df['service_impediment']
    
    # Paradox indicator: 포화 + 원활 → 고부하 효율 운영
    df['paradox_indicator'] = df['pack_utilization'].fillna(0) * (1 - df['service_impediment'])
    
    return df
```

## 파이프라인 (Phase 16 기반 재현)

### 기반 파일
- `run_phase16_fe.py`를 복사하여 `run_phase18_recovery.py` 생성
- Phase 16 구조 100% 유지
  - StratifiedGroupKFold (기존 CV 방식)
  - LGB Huber + XGB + CatBoost + TabNet
  - Part 1A (P13s1 base) + 1B (P15 agg) + 1C (P16 2nd-order)
  - 기존 692 features

### 수정 사항 (단 2곳)

**수정 1: Target transform**
```python
# 학습 직전
y_raw = train['avg_delay_minutes_next_30m'].copy()
y_transformed = np.log1p(y_raw)  # log(1+y)

# 모든 모델 학습에 y_transformed 사용
# objective는 기존 'huber' 유지 (이유: log space에서 Huber가 여전히 robust)
# LGB, XGB, CatBoost 모두 동일 objective 유지

# OOF 및 test 예측 후 역변환
def inverse_transform(pred_log):
    pred = np.expm1(pred_log)
    pred = np.clip(pred, 0, None)
    return pred

oof_pred = inverse_transform(oof_pred_log)
test_pred = inverse_transform(test_pred_log)
```

**수정 2: Composite features 추가**
```python
# Feature engineering 단계 마지막에 추가
train = add_composite_features(train)
test = add_composite_features(test)

# Feature list에 3개 추가
feature_cols = feature_cols + ['service_impediment', 'crisis_indicator', 'paradox_indicator']
```

## 그 외 모든 것은 동일

- 하이퍼파라미터: Phase 16 그대로
- CV fold 수: 5 (Phase 16 방식)
- Seed: Phase 16 그대로 (reproducibility)
- Ensemble weights: Phase 16 그대로
- Early stopping: 기존 설정

## 검증 단계

### 1차 검증: Phase 16 재현 확인
- Log1p 적용 전 한번 실행해보고 CV MAE가 Phase 16 (8.3959) 근처인지 확인
- 차이가 > 0.05 이면 재현 실패 → 중단 (원인 분석)

### 2차 검증: Log1p 효과
- Log1p 적용 후 CV MAE 측정
- 목표: CV MAE ≤ 8.40 (Phase 16 수준 유지)
- 목표: CV MAE < 8.35 면 개선 (좋음)

### 3차 검증: Composite 추가 효과
- Composite 3개 추가 후 CV MAE 측정
- 목표: CV MAE < 8.35 (추가 개선)

## 제출

### CV가 통과하면:
- 전체 Train에서 재학습 (CV 없이)
- Test 예측
- Submission 생성: `submission_phase18_recovery.csv`
- Kaggle 제출

### 제출 메모:
"Phase 18 Recovery: Phase 16 재현 + log1p target + 3 composite features"

## 커밋 & 저장 요청

실행 완료 후:

1. **Git commit**: `git add . && git commit -m "Phase 18 Recovery: H1+H3 (log1p + composite)"`
2. **Push**: `git push origin main`
3. **Kaggle 제출**
4. **`claude_results.md`에 저장**:
   - Phase 16 재현 CV
   - Log1p 후 CV
   - Composite 후 CV
   - Kaggle public score
   - 모든 핵심 수치

## 실행 환경
- Kaggle Notebook (T4 GPU x2)
- 기존 코드 베이스: `/kaggle/working/dacon/`
- 로컬 코드: `C:\dev\dacon-warehouse-delay\`

## 주의사항

❌ **하지 말 것**:
- 외부 사례 참고 (AMEX, Ventilator 등)
- 추가 feature engineering (composite 3개 외)
- 하이퍼파라미터 튜닝
- 새로운 모델 family 추가
- 복잡한 ensemble 시도

✅ **할 것**:
- Phase 16 코드 정확히 재현
- Target log1p만 변경
- Composite 3개만 추가
- CV 철저히 측정
- 모든 수치 기록

## 성공 기준

**Pass:**
- Phase 16 재현 확인 (CV 8.35~8.45)
- Log1p 후 CV 유지 또는 개선
- Composite 후 CV 개선 (또는 유지)
- Public score ≤ 9.86

**Success:**
- Public score < 9.80

**Excellent:**
- Public score < 9.70