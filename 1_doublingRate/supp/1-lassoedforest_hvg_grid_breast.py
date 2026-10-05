"""Full n_hvg x max_depth x min_samples_leaf grid search for LassoedForest alone.

Produces growth_lassoedforest_sensitivity_summary_breast.csv, read by
2-hvg_selection_justification.ipynb as "Evidence 1" for fixing N_HVG at a
cohort-specific value (see that notebook for the full justification).
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from pipeline_utils import (  # noqa: E402
    fit_predict_lassoed_forest,
    make_case1_paths,
    safe_inner_cv,
    safe_pearson,
    safe_spearman,
    select_hvg_train_only,
)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5
FAST_Q = 0.33
SLOW_Q = 0.67

N_HVG_GRID = [100, 300, 600]
MAX_DEPTH_GRID = [3, 4, 6]
MIN_SAMPLES_LEAF_GRID = [1, 2, 5]

N_ESTIMATORS = 500
MAX_FEATURES = "sqrt"
LF_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
LR_N_JOBS = 1
ADAPTIVE_LASSO_GAMMA = 1.0
ADAPTIVE_LASSO_EPS = 1e-3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Lassoed Forest parameter sensitivity for Case Study 1."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke test with one parameter setting and fewer folds.",
    )
    return parser.parse_args()


def summarize(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=0))


def evaluate_parameter_setting(
    X_df, y_raw, model_ids, n_hvg, max_depth, min_samples_leaf,
    n_estimators, n_splits, n_repeats,
):
    from sklearn.model_selection import RepeatedKFold

    rf_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    fold_records = []
    skipped_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_df), start=1):
        X_train_df = X_df.iloc[train_idx]
        X_test_df = X_df.iloc[test_idx]
        y_train_raw = y_raw[train_idx]
        y_test_raw = y_raw[test_idx]

        fast_cutoff = np.quantile(y_train_raw, FAST_Q)
        slow_cutoff = np.quantile(y_train_raw, SLOW_Q)

        train_extreme_mask = (y_train_raw <= fast_cutoff) | (y_train_raw >= slow_cutoff)
        test_extreme_mask = (y_test_raw <= fast_cutoff) | (y_test_raw >= slow_cutoff)

        X_train_ext_df = X_train_df.loc[train_extreme_mask]
        X_test_ext_df = X_test_df.loc[test_extreme_mask]
        y_train_ext = (y_train_raw[train_extreme_mask] >= slow_cutoff).astype(int)
        y_test_ext = (y_test_raw[test_extreme_mask] >= slow_cutoff).astype(int)
        y_test_raw_ext = y_test_raw[test_extreme_mask]
        test_ids_ext = model_ids[test_idx][test_extreme_mask]

        skip_reason = None
        if X_train_ext_df.shape[0] < 4:
            skip_reason = "too_few_training_extreme_samples"
        elif len(np.unique(y_train_ext)) < 2:
            skip_reason = "training_extremes_have_one_class"
        elif len(np.unique(y_test_ext)) < 2:
            skip_reason = "test_extremes_have_one_class"

        inner_cv = None if skip_reason else safe_inner_cv(y_train_ext)
        if skip_reason is None and inner_cv is None:
            skip_reason = "too_few_training_samples_for_inner_cv"

        if skip_reason is not None:
            skipped_records.append(
                {
                    "n_hvg": n_hvg,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "Fold": fold_idx,
                    "Reason": skip_reason,
                    "n_train_extreme": int(X_train_ext_df.shape[0]),
                    "n_test_extreme": int(X_test_ext_df.shape[0]),
                    "n_train_slow": int(np.sum(y_train_ext)),
                    "n_test_slow": int(np.sum(y_test_ext)),
                }
            )
            continue

        hvg = select_hvg_train_only(X_train_ext_df, n_hvg)
        X_train = X_train_ext_df[hvg].values
        X_test = X_test_ext_df[hvg].values

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train_ext)

        probabilities, ridge, lasso, selected_trees = fit_predict_lassoed_forest(
            rf, X_train, y_train_ext, X_test, inner_cv,
            cs=LF_CS, random_state=RANDOM_STATE, n_jobs=LR_N_JOBS,
            adaptive_lasso_gamma=ADAPTIVE_LASSO_GAMMA, adaptive_lasso_eps=ADAPTIVE_LASSO_EPS,
        )

        fold_records.append(
            {
                "n_hvg": n_hvg,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "Fold": fold_idx,
                "AUROC": roc_auc_score(y_test_ext, probabilities),
                "Spearman": safe_spearman(probabilities, y_test_raw_ext),
                "Pearson": safe_pearson(probabilities, y_test_raw_ext),
                "n_train_extreme": int(X_train_ext_df.shape[0]),
                "n_test_extreme": int(X_test_ext_df.shape[0]),
                "n_train_fast": int(np.sum(y_train_ext == 0)),
                "n_train_slow": int(np.sum(y_train_ext == 1)),
                "n_test_fast": int(np.sum(y_test_ext == 0)),
                "n_test_slow": int(np.sum(y_test_ext == 1)),
                "n_test_models": int(len(test_ids_ext)),
                "n_trees_total": n_estimators,
                "n_trees_selected": selected_trees,
                "ridge_C": float(ridge.C_[0]),
                "lasso_C": float(lasso.C_[0]),
            }
        )

    return fold_records, skipped_records


def build_summary(fold_df, total_folds):
    summary_records = []

    group_cols = ["n_hvg", "max_depth", "min_samples_leaf"]
    for params, sub in fold_df.groupby(group_cols, dropna=False):
        auroc_mean, auroc_sd = summarize(sub["AUROC"])
        spearman_mean, spearman_sd = summarize(sub["Spearman"])
        pearson_mean, pearson_sd = summarize(sub["Pearson"])
        selected_mean, selected_sd = summarize(sub["n_trees_selected"])

        summary_records.append(
            {
                "n_hvg": params[0],
                "max_depth": params[1],
                "min_samples_leaf": params[2],
                "n_completed_folds": int(sub["Fold"].nunique()),
                "n_expected_folds": int(total_folds),
                "mean_AUROC": auroc_mean,
                "sd_AUROC": auroc_sd,
                "mean_Spearman": spearman_mean,
                "sd_Spearman": spearman_sd,
                "mean_Pearson": pearson_mean,
                "sd_Pearson": pearson_sd,
                "mean_selected_trees": selected_mean,
                "sd_selected_trees": selected_sd,
                "median_selected_trees": float(sub["n_trees_selected"].median()),
            }
        )

    return (
        pd.DataFrame(summary_records)
        .sort_values(["mean_AUROC", "sd_AUROC"], ascending=[False, True])
        .reset_index(drop=True)
    )


def main():
    args = parse_args()

    if args.quick:
        n_hvg_grid = [100]
        max_depth_grid = [4]
        min_samples_leaf_grid = [2]
        n_estimators = 100
        n_splits = 3
        n_repeats = 1
        output_suffix = "_quick"
    else:
        n_hvg_grid = N_HVG_GRID
        max_depth_grid = MAX_DEPTH_GRID
        min_samples_leaf_grid = MIN_SAMPLES_LEAF_GRID
        n_estimators = N_ESTIMATORS
        n_splits = N_SPLITS
        n_repeats = N_REPEATS
        output_suffix = ""

    paths = make_case1_paths()
    input_file = paths["proc"] / "rna_doubling_time_merged_breast.csv"
    summary_file = paths["proc"] / f"growth_lassoedforest_sensitivity{output_suffix}_summary_breast.csv"
    fold_file = paths["proc"] / f"growth_lassoedforest_sensitivity{output_suffix}_fold_results_breast.csv"
    skipped_file = paths["proc"] / f"growth_lassoedforest_sensitivity{output_suffix}_skipped_folds_breast.csv"
    config_file = paths["proc"] / f"growth_lassoedforest_sensitivity{output_suffix}_config_breast.json"

    df = pd.read_csv(input_file, index_col=0)
    target_col = "doubling_time_days"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in {input_file}")

    y_raw = np.log1p(df[target_col].astype(float).values)
    X_df = df.drop(columns=[target_col])
    model_ids = df.index.values

    config = {
        "input_file": str(input_file),
        "target_col": target_col,
        "target_transform": "log1p",
        "positive_class": "slow",
        "quick_mode": args.quick,
        "n_models": int(df.shape[0]),
        "n_features_input": int(X_df.shape[1]),
        "n_hvg_grid": n_hvg_grid,
        "max_depth_grid": max_depth_grid,
        "min_samples_leaf_grid": min_samples_leaf_grid,
        "n_estimators": n_estimators,
        "max_features": MAX_FEATURES,
        "fast_quantile": FAST_Q,
        "slow_quantile": SLOW_Q,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": RANDOM_STATE,
        "lassoed_forest_cs": LF_CS,
        "adaptive_lasso_gamma": ADAPTIVE_LASSO_GAMMA,
        "adaptive_lasso_eps": ADAPTIVE_LASSO_EPS,
    }
    config_file.write_text(json.dumps(config, indent=2))

    parameter_grid = list(itertools.product(n_hvg_grid, max_depth_grid, min_samples_leaf_grid))
    total_folds = n_splits * n_repeats

    print(f"Input matrix: {df.shape[0]} models x {X_df.shape[1]} genes")
    print(f"Parameter settings: {len(parameter_grid)}")
    print(f"CV folds per setting: {total_folds}")
    print(f"Config saved to: {config_file}")

    all_fold_records = []
    all_skipped_records = []

    for setting_idx, (n_hvg, max_depth, min_samples_leaf) in enumerate(parameter_grid, start=1):
        print(
            f"[{setting_idx}/{len(parameter_grid)}] "
            f"n_hvg={n_hvg}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}"
        )

        fold_records, skipped_records = evaluate_parameter_setting(
            X_df=X_df, y_raw=y_raw, model_ids=model_ids,
            n_hvg=n_hvg, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            n_estimators=n_estimators, n_splits=n_splits, n_repeats=n_repeats,
        )

        all_fold_records.extend(fold_records)
        all_skipped_records.extend(skipped_records)

    fold_df = pd.DataFrame(all_fold_records)
    skipped_df = pd.DataFrame(all_skipped_records)
    if skipped_df.empty:
        skipped_df = pd.DataFrame(
            columns=[
                "n_hvg", "max_depth", "min_samples_leaf", "Fold", "Reason",
                "n_train_extreme", "n_test_extreme", "n_train_slow", "n_test_slow",
            ]
        )

    summary_df = build_summary(fold_df, total_folds)

    fold_df.to_csv(fold_file, index=False)
    skipped_df.to_csv(skipped_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    print("\n===== Top Lassoed Forest sensitivity settings =====")
    print(
        summary_df[
            ["n_hvg", "max_depth", "min_samples_leaf", "mean_AUROC", "sd_AUROC",
             "mean_selected_trees", "n_completed_folds"]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print(f"\nSaved summary: {summary_file}")
    print(f"Saved fold-level results: {fold_file}")
    print(f"Saved skipped-fold log: {skipped_file}")


if __name__ == "__main__":
    main()
