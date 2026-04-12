run_phase23_track_a.py 수정. Codex가 3개 이슈 지적:

## Fix 1 (HIGH): Step 6 layout_type target encoding 실패
- layout_type = 4 unique values
- 현재 GroupKFold(5) 시도 → ValueError
- 해결: KFold (group 제약 없음)로 변경

## Fix 2 (MEDIUM): layout_id target encoding 설계 결함
- GroupKFold(groups=layout_id)로 split하면
  각 val fold의 layout이 train fold에 없음
- → train OOF가 모두 global_mean (상수)
- → test는 full-train encoding
- → train/test feature 불일치
- 해결: layout_id target encoding 완전 제거
  (layout 구조 변수 13개가 이미 feature에 있음)

## Fix 3 (LOW): Feature selection 개선
- 현재 single fold importance 기준
- 해결: 3-fold 평균 + "모든 fold에서 zero"인 feature 식별
- 제거 기준: zero across ALL folds OR bottom 5% 평균 importance

## 수정 코드

### Step 6 수정
```python
def oof_target_encode_simple(train_df, test_df, target_col, encode_col, n_splits=5, smoothing=10):
    from sklearn.model_selection import KFold
    global_mean = train_df[target_col].mean()
    train_encoded = np.zeros(len(train_df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for tr_idx, val_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        stats = tr_fold.groupby(encode_col)[target_col].agg(['mean', 'count'])
        smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        train_encoded[val_idx] = train_df.iloc[val_idx][encode_col].map(smooth).fillna(global_mean).values
    
    stats_full = train_df.groupby(encode_col)[target_col].agg(['mean', 'count'])
    smooth_full = (stats_full['mean'] * stats_full['count'] + global_mean * smoothing) / (stats_full['count'] + smoothing)
    test_encoded = test_df[encode_col].map(smooth_full).fillna(global_mean).values
    
    return train_encoded, test_encoded

# layout_id target encoding 완전 제거
# (layout 구조 변수들로 충분)
print("  Skipping layout_id target encoding (GroupKFold design issue)")
print("  Layout info already captured via layout_info.csv features")

# layout_type만 encoding (simple KFold)
print("  Encoding layout_type...")
train['layout_type_te'], test['layout_type_te'] = oof_target_encode_simple(
    train, test, TARGET, 'layout_type', n_splits=5, smoothing=5
)
```

### Step 8 수정 (3-fold importance)
```python
print("\n" + "="*70)
print("Step 8: Multi-fold Feature Selection")
print("="*70)

drop_cols = ['ID', 'scenario_id', 'layout_id', 'layout_type', TARGET]
feature_cols = [c for c in train.columns if c not in drop_cols]
numeric_cols = [c for c in feature_cols if train[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
feature_cols = numeric_cols
print(f"  Total numeric features: {len(feature_cols)}")

X = train[feature_cols]
y = train[TARGET]
groups = train['layout_id']

# 3-fold importance (다른 seed)
importance_folds = []
gkf_imp = GroupKFold(n_splits=3)

for fold_idx, (tr_idx, val_idx) in enumerate(gkf_imp.split(X, y, groups)):
    quick_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, feature_fraction=0.7,
        objective='huber', random_state=42 + fold_idx,
        verbose=-1, n_jobs=-1
    )
    quick_model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
    )
    fold_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': quick_model.feature_importances_
    })
    importance_folds.append(fold_imp)
    print(f"  Fold {fold_idx+1} importance computed")

imp = pd.concat(importance_folds).groupby('feature')['importance'].mean().reset_index()
imp = imp.sort_values('importance', ascending=False)
imp.to_csv('output/phase23_track_a_importance.csv', index=False)

# 제거 기준 1: ALL folds에서 zero
zero_imp_all = set(feature_cols)
for fold_imp in importance_folds:
    non_zero = set(fold_imp[fold_imp['importance'] > 0]['feature'])
    zero_imp_all -= non_zero
print(f"  Zero in ALL folds: {len(zero_imp_all)}")

# 제거 기준 2: 평균 하위 5%
low_threshold = imp['importance'].quantile(0.05)
low_imp = set(imp[imp['importance'] <= low_threshold]['feature'].tolist())
print(f"  Bottom 5% avg importance: {len(low_imp)}")

# 합집합
remove_features = zero_imp_all | low_imp
final_features = [f for f in feature_cols if f not in remove_features]

print(f"\n  Removing: {len(remove_features)} features")
print(f"  Final: {len(final_features)}")
print(f"  Top 10: {imp.head(10)['feature'].tolist()}")

# Save
with open('output/phase23_track_a_features.pkl', 'wb') as f:
    pickle.dump({
        'all_features': feature_cols,
        'zero_imp_all_folds': list(zero_imp_all),
        'low_imp': list(low_imp),
        'remove_features': list(remove_features),
        'final_features': final_features,
        'importance': imp.to_dict('records')
    }, f)
```

## 기타 확인
- Step 9 (full CV)와 Step 10 (save)는 변경 없음
- layout_id_te 참조하는 다른 코드 없는지 확인
- ast.parse 통과 확인
- 커밋: "fix: Phase 23 Track A - target encoding + multi-fold selection"
- 푸시
- claude_results.md 저장