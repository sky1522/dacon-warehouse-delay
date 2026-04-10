# Checkpoint Manager

체크포인트 명명, 저장, 검증 패턴.

## 명명 규칙
```
ckpt_phase{NN}[s{seed}]_{model}.pkl
```
- `NN`: Phase 번호 (01~99)
- `s{seed}`: multi-seed 시 seed 번호 (s42, s777 등)
- `model`: lgb_raw, lgb_huber, lgb_sqrt, xgb, cat_log1p, cat_raw, mlp, tabnet

## 필수 metadata
```python
data = {
    'pipeline_version': "phase16_v1",
    'feature_cols': list(feature_cols),
    'n_features': len(feature_cols),
    'oof': oof_array,           # np.float32
    'test': test_array,         # np.float32
    'cv_mae': float(cv_mae),
    'random_seed': 42,
}
```

## save_ckpt / load_ckpt 패턴
```python
def save_ckpt(local_path, data, feature_cols=None):
    if feature_cols is not None:
        data['feature_cols'] = list(feature_cols)
        data['n_features'] = len(feature_cols)
    data['pipeline_version'] = PIPELINE_VERSION
    with open(local_path, 'wb') as f:
        pickle.dump(data, f)
    # Drive 즉시 동기화
    drive_path = os.path.join(DRIVE_CKPT_DIR, os.path.basename(local_path))
    if os.path.exists(os.path.dirname(drive_path)):
        shutil.copy(local_path, drive_path)

def load_ckpt(local_path, expected_features=None):
    for path in [drive_path, local_path]:
        if os.path.exists(path):
            ckpt = pickle.load(open(path, 'rb'))
            # Version 검증
            if ckpt.get('pipeline_version') != PIPELINE_VERSION:
                return None  # Cache invalidation
            # Feature 검증
            if expected_features and ckpt.get('feature_cols') != list(expected_features):
                return None
            return ckpt
    return None
```

## 주의
- OOF shape 검증: `len(ckpt['oof']) == len(y_train)` 필수
- Cross-phase 로딩 시 pipeline_version 불일치 → cache invalidation
