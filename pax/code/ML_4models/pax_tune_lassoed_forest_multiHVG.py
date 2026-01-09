#!/usr/bin/env python3
"""
pax_tune_lassoed_forest_multiHVG.py

Tune "Lassoed Forest" for paclitaxel response (binary pax_cat) across multiple HVG sizes.

Lassoed Forest (classification) = RandomForestClassifier -> leaf indices -> OneHotEncoder -> L1 LogisticRegression

Key points (PDX-safe):
- Stratified CV (no single split overfitting)
- HVG selection is done *inside each training fold* to avoid leakage
- Scaling is fit on training fold only
- Leaf IDs are one-hot encoded (critical; do NOT treat leaf ids as numeric)

Outputs:
- all_results.csv                 : every hyperparam combination evaluated (mean±std across folds)
- best_by_HVG.csv                 : best row per HVG (selected by AUROC, tie-break AUPRC)
- best_LassoedForest_HVG<N>.json  : detailed best config per HVG

Input:
- ../../data/procdata/RNAseq_with_pax_800gene.csv
  columns: modelID, pax_mRECIST, pax_cat, pax_ord, <genes...>

Author: you + ChatGPT
"""

from __future__ import annotations

import os
import json
import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# -----------------------------
# Config
# -----------------------------
#HVG_LIST = [300, 500, 600, 700, 800]  # you can change this

IN_FILE = "../../data/procdata/RNAseq_with_pax_800gene.csv"
OUT_DIR = "../../data/ML_4model/PAX_LassoedForest_tuning_multiHVG"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
#N_SPLITS = 5  # stratified CV folds

# RF hyperparams: regularized-ish for N~68
# RF_GRID = {
#     "n_estimators": [500, 1000, 2000],
#     "max_depth": [2, 3, 4, 5],
#     "min_samples_leaf": [2, 4, 6, 8],
#     "min_samples_split": [2, 4],
#     "max_features": ["sqrt", 0.5, 0.7],
# }

# L1 logistic regularization strength
# LogisticRegression uses C = inverse regularization
#C_LIST = np.logspace(-3, 2, 8).tolist()  # 1e-3 ... 1e2

# HVG_LIST = [300, 450, 600]
# RF_GRID = {
#     "n_estimators": [800],
#     "max_depth": [4, 5],
#     "min_samples_leaf": [2, 4],
#     "min_samples_split": [2],
#     "max_features": ["sqrt"],
# }
# C_LIST = [100, 300, 1000]
# N_SPLITS = 3

HVG_LIST = [300, 600]

RF_GRID = {
    "n_estimators": [1000, 2000],
    "max_depth": [4, 5, 6],
    "min_samples_leaf": [2],
    "min_samples_split": [2],
    "max_features": ["sqrt"],
}

C_LIST = [1000]

# Logistic regression
penalty = "elasticnet"
l1_ratio_list = [0.3, 0.5, 0.7]

N_SPLITS = 5   # important for stability now

# Tie-breaker: AUROC first, then AUPRC
PRIMARY_METRIC = "AUROC"
SECONDARY_METRIC = "AUPRC"


# -----------------------------
# Utilities
# -----------------------------
def select_hvgs(X_train_df: pd.DataFrame, n: int) -> list[str]:
    """Select top-n most variable genes using training fold only."""
    var = X_train_df.var(axis=0)
    return var.sort_values(ascending=False).head(n).index.tolist()


def metrics_binary(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a standard binary classification metric set."""
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "ACC": accuracy_score(y_true, y_pred),
        "BACC": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }


def mean_std_dict(list_of_dicts: list[dict]) -> dict:
    """Return mean and std for each numeric key across list of dicts."""
    df = pd.DataFrame(list_of_dicts)
    out = {}
    for col in df.columns:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def iter_grid(grid: dict) -> list[dict]:
    """Expand a dict-of-lists grid to a list of dict configs."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*vals):
        configs.append({k: v for k, v in zip(keys, combo)})
    return configs


# -----------------------------
# Core evaluation
# -----------------------------
def eval_config_for_hvg(
    X_full: pd.DataFrame,
    y_full: np.ndarray,
    hvg_n: int,
    rf_params: dict,
    C: float,
    *,
    random_state: int = RANDOM_STATE,
    n_splits: int = N_SPLITS,
) -> dict:
    """
    Evaluate a single (HVG, RF params, C) configuration using stratified CV.
    Returns mean±std across folds for AUROC/AUPRC/ACC/BACC/F1.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for tr_idx, te_idx in skf.split(X_full, y_full):
        Xtr_df = X_full.iloc[tr_idx]
        Xte_df = X_full.iloc[te_idx]
        ytr = y_full[tr_idx]
        yte = y_full[te_idx]

        # HVG selection on training fold ONLY (avoid leakage)
        hvgs = select_hvgs(Xtr_df, n=hvg_n)
        Xtr = Xtr_df[hvgs].values
        Xte = Xte_df[hvgs].values

        # Scale on training fold only
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        # RF to generate leaf rules
        rf = RandomForestClassifier(**rf_params, random_state=random_state, n_jobs=-1)
        rf.fit(Xtr_s, ytr)

        leaves_tr = rf.apply(Xtr_s)  # (n_train, n_trees)
        leaves_te = rf.apply(Xte_s)  # (n_test,  n_trees)

        # One-hot encode leaf IDs (CRITICAL)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        Ztr = ohe.fit_transform(leaves_tr)
        Zte = ohe.transform(leaves_te)

        # L1 logistic on sparse leaf indicators
        logit = LogisticRegression(
            penalty="l1", solver="saga", C=C, max_iter=20000, random_state=random_state
        )
        logit.fit(Ztr, ytr)

        prob = logit.predict_proba(Zte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        fold_metrics.append(metrics_binary(yte, prob, pred))

    agg = mean_std_dict(fold_metrics)

    # pack config + results
    out = {
        "HVG": hvg_n,
        "C": C,
        **{f"rf_{k}": v for k, v in rf_params.items()},
        **agg,
        "n_splits": n_splits,
    }
    return out


def pick_best(rows: list[dict]) -> dict:
    """Pick best row by PRIMARY_METRIC_mean, tie-break SECONDARY_METRIC_mean."""

    def key_fn(r):
        return (r[f"{PRIMARY_METRIC}_mean"], r[f"{SECONDARY_METRIC}_mean"])

    return max(rows, key=key_fn)


# -----------------------------
# Main
# -----------------------------
def main():
    # Load data
    df = pd.read_csv(IN_FILE, dtype={"modelID": str})
    meta_cols = ["modelID", "pax_mRECIST", "pax_cat", "pax_ord"]
    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    X_full = df.drop(columns=meta_cols)
    y_full = df["pax_cat"].astype(int).values

    print(f"Loaded: {X_full.shape[0]} samples × {X_full.shape[1]} genes")
    print("Class balance:", dict(pd.Series(y_full).value_counts().sort_index()))

    # Expand grids
    rf_configs = iter_grid(RF_GRID)

    all_rows = []
    best_rows = []

    # Tune per HVG
    for hvg_n in HVG_LIST:
        print("\n" + "=" * 78)
        print(f"🔬 Tuning Lassoed Forest for HVG={hvg_n}")
        print("=" * 78)

        rows_this_hvg = []
        total = len(rf_configs) * len(C_LIST)
        i = 0

        for rf_params in rf_configs:
            for C in C_LIST:
                i += 1
                res = eval_config_for_hvg(X_full, y_full, hvg_n, rf_params, C)
                rows_this_hvg.append(res)
                all_rows.append(res)

                print(
                    f"[HVG={hvg_n}] {i:>4}/{total} "
                    f"RF(n={rf_params['n_estimators']},depth={rf_params['max_depth']},"
                    f"leaf={rf_params['min_samples_leaf']},split={rf_params['min_samples_split']},"
                    f"feat={rf_params['max_features']}) "
                    f"C={C:.4g} | "
                    f"AUROC={res['AUROC_mean']:.3f}±{res['AUROC_std']:.3f} "
                    f"AUPRC={res['AUPRC_mean']:.3f}±{res['AUPRC_std']:.3f} "
                    f"BACC={res['BACC_mean']:.3f}"
                )

        best = pick_best(rows_this_hvg)
        best_rows.append(best)

        # Save detailed best config
        best_path = os.path.join(OUT_DIR, f"best_LassoedForest_HVG{hvg_n}.json")
        with open(best_path, "w") as f:
            json.dump(best, f, indent=2)

        print("\n🎯 Best for HVG=", hvg_n)
        print(
            f"  AUROC={best['AUROC_mean']:.4f}  AUPRC={best['AUPRC_mean']:.4f}  "
            f"BACC={best['BACC_mean']:.4f}  F1={best['F1_mean']:.4f}"
        )
        print(f"📁 Saved → {best_path}")

    # Save all results
    all_df = pd.DataFrame(all_rows)
    all_path = os.path.join(OUT_DIR, "all_results.csv")
    all_df.to_csv(all_path, index=False, float_format="%.6f")
    print("\n📁 Saved all results →", all_path)

    # Save best-by-HVG
    best_df = pd.DataFrame(best_rows).sort_values(by="HVG")
    best_path = os.path.join(OUT_DIR, "best_by_HVG.csv")
    best_df.to_csv(best_path, index=False, float_format="%.6f")
    print("📁 Saved best-by-HVG →", best_path)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
