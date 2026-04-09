"""
Phase 14: Scenario-level Bidirectional GRU
- Treat each scenario as a 25-timestep sequence
- Static (layout) + Dynamic (per-timestep) feature separation
- Multi-seed training with StratifiedGroupKFold on layout_id
- Blending with Phase 13s1
"""

import os
import random
import shutil
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

DRIVE_CKPT_DIR = '/content/drive/MyDrive/dacon_ckpt'
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# ============================================================
# 1. Data Loading
# ============================================================
print("=== Data Load ===", flush=True)
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
layout_info = pd.read_csv('data/layout_info.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')

train_df = train_df.merge(layout_info, on='layout_id', how='left')
test_df = test_df.merge(layout_info, on='layout_id', how='left')

# time_idx: use implicit_timeslot (cumcount within scenario)
for df in [train_df, test_df]:
    df['time_idx'] = df.groupby('scenario_id').cumcount()

train_df = train_df.sort_values(['scenario_id', 'time_idx']).reset_index(drop=True)
test_df = test_df.sort_values(['scenario_id', 'time_idx']).reset_index(drop=True)

# Backup original test ID order for submission
test_original_order = pd.read_csv('data/test.csv')['ID'].values

print(f"Train: {train_df.shape}, Test: {test_df.shape}", flush=True)
print(f"Train scenarios: {train_df['scenario_id'].nunique()}, Test scenarios: {test_df['scenario_id'].nunique()}", flush=True)

# ============================================================
# 2. Feature Separation + Time Encoding
# ============================================================
print("\n=== Feature Separation ===", flush=True)

# Static features from layout_info
STATIC_COLS = [
    'layout_type',
    'robot_total', 'pack_station_count', 'charger_count',
    'aisle_width_avg', 'intersection_count', 'one_way_ratio',
    'layout_compactness', 'zone_dispersion',
    'floor_area_sqm', 'ceiling_height_m', 'building_age_years',
    'fire_sprinkler_count', 'emergency_exit_count',
]

# layout_type one-hot
train_df = pd.concat([train_df, pd.get_dummies(train_df['layout_type'], prefix='lt')], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['layout_type'], prefix='lt')], axis=1)

# Align one-hot columns
lt_cols_train = set(c for c in train_df.columns if c.startswith('lt_'))
lt_cols_test = set(c for c in test_df.columns if c.startswith('lt_'))
for c in lt_cols_train - lt_cols_test:
    test_df[c] = 0
for c in lt_cols_test - lt_cols_train:
    train_df[c] = 0
lt_cols = sorted(lt_cols_train | lt_cols_test)

STATIC_NUM_COLS = [c for c in STATIC_COLS if c != 'layout_type']
STATIC_FEATURES = STATIC_NUM_COLS + lt_cols

# Dynamic features: auto-detect (std > 0 within scenarios)
EXCLUDE_COLS = set(['ID', 'scenario_id', 'time_idx', 'layout_id',
                    'layout_type', 'shift_hour', 'day_of_week',
                    'avg_delay_minutes_next_30m'] + STATIC_FEATURES)

dynamic_candidates = [c for c in train_df.columns
                      if c not in EXCLUDE_COLS
                      and train_df[c].dtype in [np.float32, np.float64, np.int64, np.int32]
                      and train_df[c].isna().mean() < 0.5]

DYNAMIC_FEATURES = dynamic_candidates

# Time feature encoding
for df in [train_df, test_df]:
    df['shift_hour'] = df['shift_hour'].fillna(-1)
    df['hour_sin'] = np.sin(2 * np.pi * df['shift_hour'] / 24).astype('float32')
    df['hour_cos'] = np.cos(2 * np.pi * df['shift_hour'] / 24).astype('float32')
    df['dow'] = df['day_of_week'].fillna(-1)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7).astype('float32')
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7).astype('float32')
    df['ts_norm'] = (df.groupby('scenario_id').cumcount() / 24.0).astype('float32')

TIME_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'ts_norm']
DYNAMIC_FEATURES_FULL = DYNAMIC_FEATURES + TIME_FEATURES

print(f"Static features: {len(STATIC_FEATURES)}", flush=True)
print(f"Dynamic features: {len(DYNAMIC_FEATURES)}", flush=True)
print(f"Dynamic + time: {len(DYNAMIC_FEATURES_FULL)}", flush=True)

# ============================================================
# 3. Reshape to sequences
# ============================================================
print("\n=== Reshape to Sequences ===", flush=True)


def reshape_to_sequences(df, static_cols, dynamic_cols, target_col=None):
    df_sorted = df.sort_values(['scenario_id', 'time_idx']).reset_index(drop=True)

    n_scenarios = df_sorted['scenario_id'].nunique()
    n_timesteps = 25

    X_dyn = df_sorted[dynamic_cols].values.reshape(n_scenarios, n_timesteps, len(dynamic_cols))
    X_stat = df_sorted.groupby('scenario_id').first()[static_cols].values
    scenario_ids = df_sorted['scenario_id'].unique()
    layout_ids = df_sorted.groupby('scenario_id').first()['layout_id'].values

    if target_col is not None:
        y = df_sorted[target_col].values.reshape(n_scenarios, n_timesteps)
        return X_stat.astype('float32'), X_dyn.astype('float32'), y.astype('float32'), scenario_ids, layout_ids
    else:
        return X_stat.astype('float32'), X_dyn.astype('float32'), None, scenario_ids, layout_ids


# Fill NaN before reshape
for col in STATIC_FEATURES + DYNAMIC_FEATURES_FULL:
    train_df[col] = train_df[col].fillna(0).astype('float32')
    test_df[col] = test_df[col].fillna(0).astype('float32')

X_stat_tr, X_dyn_tr, y_tr, scenario_ids_tr, layout_ids_tr = reshape_to_sequences(
    train_df, STATIC_FEATURES, DYNAMIC_FEATURES_FULL, 'avg_delay_minutes_next_30m'
)
X_stat_te, X_dyn_te, _, scenario_ids_te, layout_ids_te = reshape_to_sequences(
    test_df, STATIC_FEATURES, DYNAMIC_FEATURES_FULL, None
)

print(f"Train: static {X_stat_tr.shape}, dynamic {X_dyn_tr.shape}, y {y_tr.shape}", flush=True)
print(f"Test: static {X_stat_te.shape}, dynamic {X_dyn_te.shape}", flush=True)

assert X_stat_tr.shape[0] == 10000, f"Expected 10000 scenarios, got {X_stat_tr.shape[0]}"
assert X_dyn_tr.shape == (10000, 25, len(DYNAMIC_FEATURES_FULL))
assert X_stat_te.shape[0] == 2000

# ============================================================
# 4. Normalization
# ============================================================
print("\n=== Normalization ===", flush=True)

n_dyn = X_dyn_tr.shape[-1]
dyn_scaler = StandardScaler()
X_dyn_tr_flat = X_dyn_tr.reshape(-1, n_dyn)
dyn_scaler.fit(X_dyn_tr_flat)
X_dyn_tr = dyn_scaler.transform(X_dyn_tr_flat).reshape(X_dyn_tr.shape).astype('float32')
X_dyn_te = dyn_scaler.transform(X_dyn_te.reshape(-1, n_dyn)).reshape(X_dyn_te.shape).astype('float32')

stat_scaler = StandardScaler()
X_stat_tr = stat_scaler.fit_transform(X_stat_tr).astype('float32')
X_stat_te = stat_scaler.transform(X_stat_te).astype('float32')

# Clip extreme values
X_dyn_tr = np.clip(X_dyn_tr, -5, 5)
X_dyn_te = np.clip(X_dyn_te, -5, 5)
X_stat_tr = np.clip(X_stat_tr, -5, 5)
X_stat_te = np.clip(X_stat_te, -5, 5)

# Target: log1p transform
y_tr_log = np.log1p(y_tr)

print(f"Dynamic scaler fitted on {X_dyn_tr_flat.shape[0]} samples", flush=True)
print(f"Target log1p: min={y_tr_log.min():.2f}, max={y_tr_log.max():.2f}, mean={y_tr_log.mean():.2f}", flush=True)

# ============================================================
# 5. Model: Scenario GRU
# ============================================================


class ScenarioGRU(nn.Module):
    def __init__(self, n_static, n_dynamic, hidden=256, n_layers=2, dropout=0.25):
        super().__init__()

        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        self.gru = nn.GRU(
            input_size=n_dynamic,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_static, x_dynamic):
        B, T, _ = x_dynamic.shape

        s = self.static_encoder(x_static)  # (B, hidden)
        h, _ = self.gru(x_dynamic)  # (B, 25, hidden*2)

        s_expanded = s.unsqueeze(1).expand(-1, T, -1)  # (B, 25, hidden)
        combined = torch.cat([h, s_expanded], dim=-1)  # (B, 25, hidden*3)

        out = self.head(combined).squeeze(-1)  # (B, 25)
        return out


# ============================================================
# 6. Training loop
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== GRU Training (Multi-seed) ===", flush=True)
print("=" * 60, flush=True)

# CV: scenario-level, stratified by target mean, grouped by layout
scenario_y_mean = y_tr.mean(axis=1)  # (10000,)
y_binned_scenario = pd.qcut(scenario_y_mean, q=5, labels=False, duplicates='drop')

SEEDS = [42, 2024, 777]
BATCH_SIZE = 256
N_EPOCHS = 30
MAX_PATIENCE = 5


def train_one_seed(seed):
    set_seed(seed)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_pred = np.zeros_like(y_tr)  # (10000, 25)
    test_pred = np.zeros((X_stat_te.shape[0], 25), dtype='float32')

    fold_maes = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        cv.split(np.arange(len(scenario_y_mean)), y_binned_scenario, groups=layout_ids_tr)
    ):
        print(f"\n  [Seed {seed}] Fold {fold_idx+1}/5 (train={len(tr_idx)}, val={len(va_idx)})...", flush=True)

        # Verify no layout overlap
        tr_layouts = set(layout_ids_tr[tr_idx])
        va_layouts = set(layout_ids_tr[va_idx])
        assert len(tr_layouts & va_layouts) == 0, "Layout overlap in fold!"

        X_stat_tr_f = torch.tensor(X_stat_tr[tr_idx]).to(device)
        X_dyn_tr_f = torch.tensor(X_dyn_tr[tr_idx]).to(device)
        y_tr_f = torch.tensor(y_tr_log[tr_idx]).to(device)

        X_stat_va_f = torch.tensor(X_stat_tr[va_idx]).to(device)
        X_dyn_va_f = torch.tensor(X_dyn_tr[va_idx]).to(device)
        y_va_raw = y_tr[va_idx]

        X_stat_te_t = torch.tensor(X_stat_te).to(device)
        X_dyn_te_t = torch.tensor(X_dyn_te).to(device)

        model = ScenarioGRU(
            n_static=X_stat_tr.shape[1],
            n_dynamic=X_dyn_tr.shape[2],
            hidden=256, n_layers=2, dropout=0.25,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
        criterion = nn.L1Loss()

        best_val_mae = float('inf')
        best_oof_pred = None
        best_test_pred = None
        patience = 0

        for epoch in range(N_EPOCHS):
            # Train
            model.train()
            perm = torch.randperm(len(tr_idx))
            train_loss = 0
            n_batches = 0

            for start in range(0, len(tr_idx), BATCH_SIZE):
                batch_idx = perm[start:start + BATCH_SIZE]
                xs = X_stat_tr_f[batch_idx]
                xd = X_dyn_tr_f[batch_idx]
                yb = y_tr_f[batch_idx]

                optimizer.zero_grad()
                pred = model(xs, xd)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred_log_list = []
                for start in range(0, len(va_idx), BATCH_SIZE):
                    xs = X_stat_va_f[start:start + BATCH_SIZE]
                    xd = X_dyn_va_f[start:start + BATCH_SIZE]
                    p = model(xs, xd)
                    val_pred_log_list.append(p.cpu().numpy())
                val_pred_log = np.concatenate(val_pred_log_list, axis=0)
                val_pred = np.clip(np.expm1(np.clip(val_pred_log, 0, 10)), 0, None).astype('float32')
                val_mae = np.abs(val_pred.flatten() - y_va_raw.flatten()).mean()

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_oof_pred = val_pred
                patience = 0

                # Test prediction at best epoch
                with torch.no_grad():
                    test_pred_list = []
                    for start in range(0, X_stat_te_t.shape[0], BATCH_SIZE):
                        xs = X_stat_te_t[start:start + BATCH_SIZE]
                        xd = X_dyn_te_t[start:start + BATCH_SIZE]
                        p = model(xs, xd)
                        test_pred_list.append(p.cpu().numpy())
                    best_test_pred = np.concatenate(test_pred_list, axis=0)
            else:
                patience += 1
                if patience >= MAX_PATIENCE:
                    print(f"    Early stop at epoch {epoch+1}, best MAE: {best_val_mae:.4f}", flush=True)
                    break

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: train_loss={train_loss/n_batches:.4f}, val_MAE={val_mae:.4f}", flush=True)

        oof_pred[va_idx] = best_oof_pred
        # Inverse log1p for test predictions
        best_test_pred_raw = np.clip(np.expm1(np.clip(best_test_pred, 0, 10)), 0, None).astype('float32')
        test_pred += best_test_pred_raw / 5
        fold_maes.append(best_val_mae)
        print(f"    Fold {fold_idx+1} best MAE: {best_val_mae:.4f}", flush=True)

        # Free GPU memory
        del model, optimizer, scheduler
        del X_stat_tr_f, X_dyn_tr_f, y_tr_f, X_stat_va_f, X_dyn_va_f
        del X_stat_te_t, X_dyn_te_t
        torch.cuda.empty_cache()

    cv_mae = np.abs(oof_pred.flatten() - y_tr.flatten()).mean()
    print(f"  Seed {seed} CV MAE: {cv_mae:.4f} (fold MAEs: {[f'{m:.4f}' for m in fold_maes]})", flush=True)

    # Save checkpoint
    ckpt_path = f'output/ckpt_phase14_gru_seed{seed}.pt'
    torch.save({'oof': oof_pred, 'test': test_pred, 'cv_mae': cv_mae}, ckpt_path)
    drive_path = os.path.join(DRIVE_CKPT_DIR, os.path.basename(ckpt_path))
    shutil.copy(ckpt_path, drive_path)
    print(f"  Saved: {ckpt_path} + {drive_path}", flush=True)

    return oof_pred, test_pred


# Multi-seed training
all_oof = []
all_test = []

for seed in SEEDS:
    print(f"\n{'='*60}", flush=True)
    print(f"=== Training seed {seed} ===", flush=True)
    print(f"{'='*60}", flush=True)
    oof_s, test_s = train_one_seed(seed)
    all_oof.append(oof_s)
    all_test.append(test_s)

# Average across seeds
oof_gru = np.mean(all_oof, axis=0)  # (10000, 25)
test_gru = np.mean(all_test, axis=0)  # (2000, 25)

# Flatten to row format
oof_gru_flat = oof_gru.flatten()  # (250000,)
test_gru_flat = test_gru.flatten()  # (50000,)

cv_mae = np.abs(oof_gru_flat - y_tr.flatten()).mean()
print(f"\n=== GRU Multi-seed CV MAE: {cv_mae:.4f} ===", flush=True)

# ============================================================
# 7. Save OOF / Submission
# ============================================================
print("\n=== Save OOF and Submission ===", flush=True)

# OOF for future blending
gru_oof_df = pd.DataFrame({
    'scenario_id': np.repeat(scenario_ids_tr, 25),
    'time_idx_in_scenario': np.tile(np.arange(25), len(scenario_ids_tr)),
    'gru_pred': oof_gru_flat,
    'y': y_tr.flatten(),
})
gru_oof_df.to_csv('output/phase14_gru_oof.csv', index=False)
print(f"  OOF saved: output/phase14_gru_oof.csv", flush=True)

# Reorder test to original ID order
test_scenario_order = test_df.sort_values(['scenario_id', 'time_idx'])['ID'].values
submission = pd.DataFrame({
    'ID': test_scenario_order,
    'avg_delay_minutes_next_30m': test_gru_flat,
})

# Restore original test ID order
submission = submission.set_index('ID').reindex(sample_sub['ID']).reset_index()
submission['avg_delay_minutes_next_30m'] = submission['avg_delay_minutes_next_30m'].clip(0, 500)

assert (submission['ID'].values == sample_sub['ID'].values).all(), "ID order mismatch!"
assert len(submission) == len(sample_sub), "Row count mismatch!"
assert (submission['avg_delay_minutes_next_30m'] >= 0).all(), "Negative predictions!"

submission.to_csv('output/submission_phase14_gru.csv', index=False)
shutil.copy('output/submission_phase14_gru.csv',
            os.path.join(DRIVE_CKPT_DIR, 'submission_phase14_gru.csv'))

print(f"submission_phase14_gru.csv saved", flush=True)
print(f"\nPrediction stats:", flush=True)
print(submission['avg_delay_minutes_next_30m'].describe(), flush=True)

# ============================================================
# 8. Blending with Phase 13s1
# ============================================================
print("\n=== Blending with Phase 13s1 ===", flush=True)

s13_path_drive = os.path.join(DRIVE_CKPT_DIR, 'submission_phase13s1.csv')
s13_path_local = 'output/submission_phase13s1.csv'

s13 = None
if os.path.exists(s13_path_drive):
    s13 = pd.read_csv(s13_path_drive)
    print(f"  Loaded Phase 13s1 from Drive", flush=True)
elif os.path.exists(s13_path_local):
    s13 = pd.read_csv(s13_path_local)
    print(f"  Loaded Phase 13s1 from local", flush=True)

if s13 is not None:
    for w_gru in [0.2, 0.3, 0.4, 0.5]:
        blend = submission.copy()
        blend['avg_delay_minutes_next_30m'] = (
            (1 - w_gru) * s13['avg_delay_minutes_next_30m'] +
            w_gru * submission['avg_delay_minutes_next_30m']
        ).clip(0, 500)
        blend_name = f'submission_phase14_blend_{int(w_gru*100)}.csv'
        blend_path = f'output/{blend_name}'
        blend.to_csv(blend_path, index=False)
        shutil.copy(blend_path, os.path.join(DRIVE_CKPT_DIR, blend_name))
        print(f"  Blend gru={w_gru}: {blend_path}", flush=True)
else:
    print("  Phase 13s1 submission not found, skipping blend", flush=True)

# ============================================================
# 9. Results Summary
# ============================================================
print("\n" + "=" * 60, flush=True)
print("=== Phase 14 GRU Results ===", flush=True)
print("=" * 60, flush=True)
print(f"Model: Bidirectional GRU (hidden=256, layers=2, dropout=0.25)", flush=True)
print(f"Static features: {len(STATIC_FEATURES)}", flush=True)
print(f"Dynamic features: {len(DYNAMIC_FEATURES_FULL)}", flush=True)
print(f"Seeds: {SEEDS}", flush=True)
print(f"CV: StratifiedGroupKFold(layout_id, target_bin=5)", flush=True)
print(f"Multi-seed CV MAE: {cv_mae:.4f}", flush=True)
print(f"Phase 13s1 baseline: 8.5668", flush=True)
print(f"Difference: {cv_mae - 8.5668:+.4f}", flush=True)

print("\n=== Phase 14 Complete ===", flush=True)
