import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "avg_delay_minutes_next_30m"
TS_COLS = [
    "battery_mean",
    "low_battery_ratio",
    "order_inflow_15m",
    "congestion_score",
    "robot_idle",
    "robot_charging",
    "max_zone_density",
    "pack_utilization",
]
CORE_PROFILE_COLS = [
    "battery_mean",
    "order_inflow_15m",
    "congestion_score",
    "low_battery_ratio",
    "robot_idle",
    "robot_charging",
    "max_zone_density",
    "pack_utilization",
    "order_per_active_robot",
    "battery_bottleneck",
]
INTERACTION_CANDIDATES = {
    "demand_congestion": lambda df: df["order_inflow_15m"] * df["congestion_score"],
    "demand_density": lambda df: df["order_inflow_15m"] * df["max_zone_density"],
    "battery_congestion": lambda df: df["low_battery_ratio"] * df["congestion_score"],
    "charging_pressure": lambda df: df["low_battery_ratio"] * (df["charge_queue_length"] + df["robot_charging"]),
    "available_robot_buffer": lambda df: df["robot_idle"] / (df["order_inflow_15m"] + 1.0),
    "pack_dock_pressure": lambda df: df["pack_utilization"] * df["loading_dock_util"],
    "dock_wait_pressure": lambda df: df["loading_dock_util"] * df["outbound_truck_wait_min"],
    "traffic_recovery_pressure": lambda df: df["aisle_traffic_score"] * df["avg_recovery_time"],
    "collision_congestion": lambda df: df["congestion_score"] * df["near_collision_15m"],
    "battery_trip_pressure": lambda df: df["low_battery_ratio"] * df["avg_trip_distance"],
    "storage_density_congestion": lambda df: df["storage_density_pct"] * df["max_zone_density"],
    "queue_per_charger": lambda df: df["charge_queue_length"] / (df["charger_count"] + 1.0),
    "orders_per_packstation": lambda df: df["order_inflow_15m"] / (df["pack_station_count"] + 1.0),
    "robot_idle_ratio": lambda df: df["robot_idle"] / (df["robot_total"] + 1.0),
    "congestion_wait_pressure": lambda df: df["congestion_score"] * df["intersection_wait_time_avg"],
    "shift_load_pressure": lambda df: df["prev_shift_volume"] * df["order_inflow_15m"],
}


def md_table(headers, rows):
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = []
        for cell in row:
            text = str(cell).replace("|", "\\|").replace("\n", "<br>")
            cells.append(text)
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)


def feature_engineering():
    train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
    test = pd.read_csv(DATA_DIR / "test.csv", low_memory=False)
    layout = pd.read_csv(DATA_DIR / "layout_info.csv", low_memory=False)

    train["_is_train"] = 1
    test["_is_train"] = 0
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined["_original_idx"] = np.arange(len(combined))
    combined["implicit_timeslot"] = combined.groupby("scenario_id").cumcount()
    combined = combined.sort_values(["scenario_id", "implicit_timeslot"]).reset_index(drop=True)

    grouped = combined.groupby("scenario_id", sort=False)
    new_cols = {}

    for col in TS_COLS:
        col_group = grouped[col]
        lag1 = col_group.shift(1)
        lag2 = col_group.shift(2)
        lag3 = col_group.shift(3)
        lag4 = col_group.shift(4)
        lag5 = col_group.shift(5)

        new_cols[f"{col}_lag1"] = lag1
        new_cols[f"{col}_lag2"] = lag2
        new_cols[f"{col}_lag3"] = lag3
        new_cols[f"{col}_roll3_mean"] = pd.concat([lag1, lag2, lag3], axis=1).mean(axis=1)
        new_cols[f"{col}_roll3_std"] = pd.concat([lag1, lag2, lag3], axis=1).std(axis=1)
        new_cols[f"{col}_roll5_mean"] = pd.concat([lag1, lag2, lag3, lag4, lag5], axis=1).mean(axis=1)
        new_cols[f"{col}_roll5_std"] = pd.concat([lag1, lag2, lag3, lag4, lag5], axis=1).std(axis=1)

    for col in ["battery_mean", "order_inflow_15m", "congestion_score", "robot_idle"]:
        new_cols[f"{col}_diff1"] = combined[col] - new_cols[f"{col}_lag1"]

    for col in ["order_inflow_15m", "fault_count_15m", "near_collision_15m", "blocked_path_15m"]:
        new_cols[f"{col}_cumsum"] = grouped[col].cumsum()

    combined = pd.concat([combined, pd.DataFrame(new_cols)], axis=1)
    combined = combined.merge(layout, on="layout_id", how="left")

    layout_type_map = {t: i for i, t in enumerate(layout["layout_type"].unique())}
    base_extra = {
        "layout_type_encoded": combined["layout_type"].map(layout_type_map).fillna(-1).astype(int),
        "robot_per_area": combined["robot_total"] / combined["floor_area_sqm"].replace(0, np.nan),
        "charger_per_robot": combined["charger_count"] / combined["robot_total"].replace(0, np.nan),
        "packstation_per_robot": combined["pack_station_count"] / combined["robot_total"].replace(0, np.nan),
        "order_per_active_robot": combined["order_inflow_15m"] / combined["robot_active"].replace(0, np.nan),
        "sku_per_packstation": combined["unique_sku_15m"] / combined["pack_station_count"].replace(0, np.nan),
        "battery_bottleneck": combined["low_battery_ratio"] * combined["charge_queue_length"],
        "battery_spread": combined["battery_mean"] - combined["battery_std"],
    }
    combined = pd.concat([combined, pd.DataFrame(base_extra)], axis=1)

    train_part = combined[combined["_is_train"] == 1]
    missing_counts = train_part.isna().sum().sort_values(ascending=False)
    top10_missing = [c for c in missing_counts[missing_counts > 0].head(10).index if not c.startswith("_")]
    missing_flag_cols = {}
    for col in top10_missing:
        missing_flag_cols[f"{col}_missing"] = combined[col].isna().astype(int)
    if missing_flag_cols:
        combined = pd.concat([combined, pd.DataFrame(missing_flag_cols)], axis=1)

    combined = combined.sort_values("_original_idx").reset_index(drop=True)
    train_fe = combined[combined["_is_train"] == 1].copy()
    test_fe = combined[combined["_is_train"] == 0].copy()

    drop_cols = [
        "ID",
        "layout_id",
        "scenario_id",
        "layout_type",
        TARGET,
        "_is_train",
        "_original_idx",
    ]
    feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
    return train_fe, test_fe, layout, feature_cols


def train_lgb_cv(train_fe, feature_cols, group_col):
    X = train_fe[feature_cols]
    y = train_fe[TARGET]
    y_log = np.log1p(y)
    groups = train_fe[group_col]

    folds = list(GroupKFold(n_splits=5).split(X, y_log, groups))
    oof = np.zeros(len(train_fe))
    fold_rows = []
    seen_layout_rows = []

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        Xtr = X.iloc[tr_idx]
        Xva = X.iloc[va_idx]
        ytr = y_log.iloc[tr_idx]
        yva = y_log.iloc[va_idx]

        model = lgb.LGBMRegressor(
            objective="mae",
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            Xtr,
            ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred = np.clip(np.expm1(model.predict(Xva)), 0, None)
        oof[va_idx] = pred

        fold_mae = mean_absolute_error(train_fe.iloc[va_idx][TARGET], pred)
        fold_rows.append(
            {
                "fold": fold,
                "mae": float(fold_mae),
                "best_iteration": int(model.best_iteration_),
                "rows": int(len(va_idx)),
            }
        )

        train_layouts = set(train_fe.iloc[tr_idx]["layout_id"])
        val_layouts = set(train_fe.iloc[va_idx]["layout_id"])
        seen_layout_rows.append(
            {
                "fold": fold,
                "val_layouts": int(len(val_layouts)),
                "unseen_layouts": int(len(val_layouts - train_layouts)),
                "unseen_layout_ratio": float(len(val_layouts - train_layouts) / len(val_layouts)),
            }
        )

    return {
        "oof_pred": oof,
        "mae": float(mean_absolute_error(train_fe[TARGET], oof)),
        "fold_rows": fold_rows,
        "seen_layout_rows": seen_layout_rows,
    }


def series_corr(a, b):
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 100:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def build_analysis(train_fe, oof_pred, seen_cv, unseen_cv, feature_cols):
    analysis_df = train_fe[
        [
            "ID",
            "layout_id",
            "scenario_id",
            "layout_type",
            "implicit_timeslot",
            TARGET,
        ]
        + [c for c in CORE_PROFILE_COLS if c in train_fe.columns]
    ].copy()
    analysis_df["pred_lgb"] = oof_pred
    analysis_df["residual"] = analysis_df[TARGET] - analysis_df["pred_lgb"]
    analysis_df["abs_residual"] = analysis_df["residual"].abs()
    analysis_df["high_residual"] = analysis_df["abs_residual"] >= analysis_df["abs_residual"].quantile(0.95)

    high_df = analysis_df[analysis_df["high_residual"]].copy()
    high_threshold = float(analysis_df["abs_residual"].quantile(0.95))
    under_rate_high = float((high_df["residual"] > 0).mean())
    under_rate_all = float((analysis_df["residual"] > 0).mean())

    slot_mae = (
        analysis_df.groupby("implicit_timeslot")
        .apply(
            lambda g: pd.Series(
                {
                    "rows": len(g),
                    "mae": mean_absolute_error(g[TARGET], g["pred_lgb"]),
                    "mean_residual": g["residual"].mean(),
                    "high_residual_ratio": g["high_residual"].mean(),
                }
            )
        )
        .reset_index()
        .sort_values("implicit_timeslot")
    )

    slot_concentration = slot_mae.copy()
    slot_concentration["overall_slot_share"] = slot_concentration["rows"] / len(analysis_df)
    slot_concentration["high_slot_rows"] = analysis_df.groupby("implicit_timeslot")["high_residual"].sum().values
    slot_concentration["high_slot_share"] = slot_concentration["high_slot_rows"] / len(high_df)
    slot_concentration["lift"] = slot_concentration["high_slot_share"] / slot_concentration["overall_slot_share"]
    slot_concentration = slot_concentration.sort_values(["lift", "mae"], ascending=[False, False])

    layout_type_summary = (
        analysis_df.groupby("layout_type")
        .apply(
            lambda g: pd.Series(
                {
                    "rows": len(g),
                    "mae": mean_absolute_error(g[TARGET], g["pred_lgb"]),
                    "mean_residual": g["residual"].mean(),
                    "high_residual_ratio": g["high_residual"].mean(),
                }
            )
        )
        .reset_index()
    )
    layout_type_summary["overall_share"] = layout_type_summary["rows"] / len(analysis_df)
    high_layout_counts = high_df["layout_type"].value_counts(normalize=False)
    layout_type_summary["high_rows"] = layout_type_summary["layout_type"].map(high_layout_counts).fillna(0).astype(int)
    layout_type_summary["high_share"] = layout_type_summary["high_rows"] / len(high_df)
    layout_type_summary["lift"] = layout_type_summary["high_share"] / layout_type_summary["overall_share"]
    layout_type_summary = layout_type_summary.sort_values(["lift", "mae"], ascending=[False, False])

    bins = [-1e-9, 5, 20, 50, 100, np.inf]
    labels = ["0-5", "5-20", "20-50", "50-100", "100+"]
    analysis_df["target_bin"] = pd.cut(analysis_df[TARGET], bins=bins, labels=labels, right=False)
    target_bin_summary = (
        analysis_df.groupby("target_bin", observed=False)
        .apply(
            lambda g: pd.Series(
                {
                    "rows": len(g),
                    "share": len(g) / len(analysis_df),
                    "mae": mean_absolute_error(g[TARGET], g["pred_lgb"]),
                    "mean_residual": g["residual"].mean(),
                    "median_target": g[TARGET].median(),
                    "high_residual_ratio": g["high_residual"].mean(),
                }
            )
        )
        .reset_index()
    )

    feature_profile_rows = []
    for col in CORE_PROFILE_COLS:
        overall = analysis_df[col]
        high = high_df[col]
        overall_mean = overall.mean()
        high_mean = high.mean()
        std = overall.std()
        z_gap = np.nan if pd.isna(std) or std == 0 else (high_mean - overall_mean) / std
        feature_profile_rows.append(
            {
                "feature": col,
                "overall_mean": float(overall_mean),
                "high_mean": float(high_mean),
                "delta": float(high_mean - overall_mean),
                "z_gap": float(z_gap) if pd.notna(z_gap) else np.nan,
                "missing_rate_overall": float(overall.isna().mean()),
                "missing_rate_high": float(high.isna().mean()),
            }
        )
    feature_profile = pd.DataFrame(feature_profile_rows).sort_values("z_gap", ascending=False)

    interaction_rows = []
    abs_residual = analysis_df["abs_residual"]
    for name, fn in INTERACTION_CANDIDATES.items():
        series = fn(train_fe)
        corr_val = series_corr(series, abs_residual)
        overall_mean = series.mean()
        high_mean = series[analysis_df["high_residual"].values].mean()
        std = series.std()
        z_gap = np.nan if pd.isna(std) or std == 0 else (high_mean - overall_mean) / std
        interaction_rows.append(
            {
                "interaction": name,
                "corr_abs_residual": corr_val,
                "overall_mean": float(overall_mean) if pd.notna(overall_mean) else np.nan,
                "high_mean": float(high_mean) if pd.notna(high_mean) else np.nan,
                "z_gap": float(z_gap) if pd.notna(z_gap) else np.nan,
            }
        )
    interaction_summary = (
        pd.DataFrame(interaction_rows)
        .dropna(subset=["corr_abs_residual"])
        .sort_values(["corr_abs_residual", "z_gap"], ascending=[False, False])
    )

    result = {
        "feature_count": int(len(feature_cols)),
        "seen_cv_mae": float(seen_cv["mae"]),
        "unseen_cv_mae": float(unseen_cv["mae"]),
        "unseen_gap": float(unseen_cv["mae"] - seen_cv["mae"]),
        "high_residual_threshold": high_threshold,
        "high_residual_rows": int(high_df.shape[0]),
        "high_residual_ratio": float(high_df.shape[0] / analysis_df.shape[0]),
        "underprediction_rate_all": under_rate_all,
        "underprediction_rate_high": under_rate_high,
        "timeslot_mae": slot_mae.to_dict(orient="records"),
        "timeslot_concentration_top10": slot_concentration.head(10).to_dict(orient="records"),
        "layout_type_summary": layout_type_summary.to_dict(orient="records"),
        "target_bin_summary": target_bin_summary.to_dict(orient="records"),
        "feature_profile_top10": feature_profile.head(10).to_dict(orient="records"),
        "interaction_summary_top10": interaction_summary.head(10).to_dict(orient="records"),
        "seen_fold_rows": seen_cv["fold_rows"],
        "unseen_fold_rows": unseen_cv["fold_rows"],
        "seen_layout_rows": seen_cv["seen_layout_rows"],
        "unseen_layout_rows": unseen_cv["seen_layout_rows"],
    }
    return result


def write_outputs(result):
    json_path = OUTPUT_DIR / "phase3b_analysis.json"
    md_path = OUTPUT_DIR / "phase3b_preanalysis.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    timeslot_top = result["timeslot_concentration_top10"][:8]
    layout_summary = result["layout_type_summary"]
    target_bins = result["target_bin_summary"]
    feature_profile = result["feature_profile_top10"][:8]
    interactions = result["interaction_summary_top10"][:8]
    seen_folds = result["seen_fold_rows"]
    unseen_folds = result["unseen_fold_rows"]

    lines = []
    lines.append("## Phase 3B 사전 분석")
    lines.append("")
    lines.append("### 요약")
    lines.append(
        f"- LightGBM Phase 2 feature set (`{result['feature_count']}` features) gives seen-layout CV MAE `{result['seen_cv_mae']:.4f}` and unseen-layout CV MAE `{result['unseen_cv_mae']:.4f}`."
    )
    lines.append(
        f"- Simulated unseen-layout penalty is `{result['unseen_gap']:+.4f}` MAE, which can be compared directly against the observed CV-public gap of about `+1.37`."
    )
    lines.append(
        f"- Absolute residual top 5% starts at `{result['high_residual_threshold']:.4f}` minutes; those rows are `{result['high_residual_rows']:,}` samples."
    )
    lines.append(
        f"- High-residual rows are under-predictions `{result['underprediction_rate_high']:.2%}` of the time, versus `{result['underprediction_rate_all']:.2%}` across all rows."
    )
    lines.append("")
    lines.append("### Seen vs Unseen Layout MAE")
    lines.append(
        md_table(
            ["Setting", "Overall MAE", "Gap vs seen"],
            [
                ["Scenario GroupKFold (seen-layout heavy)", f"{result['seen_cv_mae']:.4f}", "0.0000"],
                ["Layout GroupKFold (unseen layout simulation)", f"{result['unseen_cv_mae']:.4f}", f"{result['unseen_gap']:+.4f}"],
            ],
        )
    )
    lines.append("")
    lines.append("### Fold-Level MAE")
    lines.append(
        md_table(
            ["Fold", "Seen CV MAE", "Unseen-layout CV MAE"],
            [
                [row_seen["fold"], f"{row_seen['mae']:.4f}", f"{row_unseen['mae']:.4f}"]
                for row_seen, row_unseen in zip(seen_folds, unseen_folds)
            ],
        )
    )
    lines.append("")
    lines.append("### Residual Hotspots by Implicit Timeslot")
    lines.append(
        md_table(
            ["Timeslot", "MAE", "Mean residual", "High-residual share", "Lift"],
            [
                [
                    int(row["implicit_timeslot"]),
                    f"{row['mae']:.4f}",
                    f"{row['mean_residual']:.4f}",
                    f"{row['high_residual_ratio']:.2%}",
                    f"{row['lift']:.2f}",
                ]
                for row in timeslot_top
            ],
        )
    )
    lines.append("")
    lines.append("### Residual Hotspots by layout_type")
    lines.append(
        md_table(
            ["layout_type", "Rows", "MAE", "Mean residual", "High-residual share", "Lift"],
            [
                [
                    row["layout_type"],
                    int(row["rows"]),
                    f"{row['mae']:.4f}",
                    f"{row['mean_residual']:.4f}",
                    f"{row['high_residual_ratio']:.2%}",
                    f"{row['lift']:.2f}",
                ]
                for row in layout_summary
            ],
        )
    )
    lines.append("")
    lines.append("### Target-Bin Accuracy")
    lines.append(
        md_table(
            ["Target bin", "Rows", "Share", "MAE", "Mean residual", "High-residual share"],
            [
                [
                    row["target_bin"],
                    int(row["rows"]),
                    f"{row['share']:.2%}",
                    f"{row['mae']:.4f}",
                    f"{row['mean_residual']:.4f}",
                    f"{row['high_residual_ratio']:.2%}",
                ]
                for row in target_bins
            ],
        )
    )
    lines.append("")
    lines.append("### Core Feature Shifts in Top-5% Residual Rows")
    lines.append(
        md_table(
            ["Feature", "Overall mean", "High-residual mean", "Delta", "Z-gap"],
            [
                [
                    row["feature"],
                    f"{row['overall_mean']:.4f}",
                    f"{row['high_mean']:.4f}",
                    f"{row['delta']:.4f}",
                    f"{row['z_gap']:.2f}",
                ]
                for row in feature_profile
            ],
        )
    )
    lines.append("")
    lines.append("### Candidate New Interactions vs Absolute Residual")
    lines.append(
        md_table(
            ["Interaction", "Corr(abs residual)", "Overall mean", "High-residual mean", "Z-gap"],
            [
                [
                    row["interaction"],
                    f"{row['corr_abs_residual']:.4f}",
                    f"{row['overall_mean']:.4f}",
                    f"{row['high_mean']:.4f}",
                    f"{row['z_gap']:.2f}",
                ]
                for row in interactions
            ],
        )
    )
    lines.append("")
    lines.append("### Recommendations")
    lines.append(
        "- Validate all future tuning on both `scenario_id` GroupKFold and a layout-holdout split. The latter is the closer proxy for public leaderboard risk."
    )
    lines.append(
        "- Add explicit interactions around demand x congestion, charge pressure, dock pressure, and storage-density congestion before wider hyperparameter sweeps."
    )
    lines.append(
        "- Give extra weight to late implicit timeslots and high-delay bins, because those regions dominate the worst residuals and are the most likely source of public-score degradation."
    )
    lines.append(
        "- In Optuna, prioritize regularization and generalization knobs first: `feature_fraction`, `bagging_fraction`, `bagging_freq`, `min_child_samples`, `num_leaves`, `lambda_l1`, `lambda_l2`, and optionally `min_gain_to_split`."
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    print("Building Phase 2 feature set...", flush=True)
    train_fe, _, _, feature_cols = feature_engineering()
    print(f"Feature count: {len(feature_cols)}", flush=True)

    print("Running seen-layout-heavy CV (group=scenario_id)...", flush=True)
    seen_cv = train_lgb_cv(train_fe, feature_cols, group_col="scenario_id")
    print(f"Seen CV MAE: {seen_cv['mae']:.4f}", flush=True)

    print("Running unseen-layout simulation (group=layout_id)...", flush=True)
    unseen_cv = train_lgb_cv(train_fe, feature_cols, group_col="layout_id")
    print(f"Unseen-layout CV MAE: {unseen_cv['mae']:.4f}", flush=True)

    print("Summarizing residual patterns...", flush=True)
    result = build_analysis(train_fe, seen_cv["oof_pred"], seen_cv, unseen_cv, feature_cols)
    json_path, md_path = write_outputs(result)

    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
