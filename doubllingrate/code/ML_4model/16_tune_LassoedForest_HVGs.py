#!/usr/bin/env python3
"""
16_tune_LassoedForest_multiHVG.py

Grid search tuning for Lassoed Forest at multiple HVG settings.
For each HVG value:
  - select top-N most variable genes
  - train RF + Lasso on log(DT+1)
  - pick best combo by Test Spearman

Outputs:
  - best_LassoedForest_by_HVG.csv  (one row per HVG)
  - best_LassoedForest_HVG<N>.txt  (detailed dict per HVG)
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
HVG_LIST = [500, 600, 700, 800]

in_file = "../../data/analyze/RNAseq_with_doublingrate_800gene.csv"
out_dir = "../../data/ML_4model/16_tuning_multiHVG"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# Load base data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X_full = df.drop(columns=meta_cols)
y_full = np.log1p(df["final_DT_mean"].astype(float))

print(f"Loaded base matrix: {X_full.shape[0]} samples × {X_full.shape[1]} genes")

# ------------------------------------------------------------
# Parameter grids (same as your single-HVG tuner)
# ------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [500, 1000],
    "max_depth": [4, 6],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 4],
    "max_features": ["sqrt"],
}

lasso_alphas = np.logspace(-3, 1, 8)   # 0.001 to 10

# ------------------------------------------------------------
# Helper: evaluate RF + Lasso on a single HVG subset
# ------------------------------------------------------------
def tune_for_hvg(n_genes: int):
    print("\n" + "=" * 70)
    print(f"🔬 Tuning Lassoed Forest for top {n_genes} HVGs")
    print("=" * 70)

    # --- select top-N most variable genes
    var = X_full.var(axis=0)
    top_genes = var.sort_values(ascending=False).head(n_genes).index.tolist()
    X = X_full[top_genes].copy()
    y = y_full.copy()

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    best = None

    def eval_combo(rf_params, alpha):
        rf = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **rf_params
        )
        rf.fit(X_train, y_train)

        # leaf encodings
        rf_train_leaves = rf.apply(X_train)
        rf_test_leaves = rf.apply(X_test)

        lasso = Lasso(alpha=alpha, max_iter=50000)
        lasso.fit(rf_train_leaves, y_train)

        y_pred_train = lasso.predict(rf_train_leaves)
        y_pred_test = lasso.predict(rf_test_leaves)

        return {
            "Train_R2": r2_score(y_train, y_pred_train),
            "Test_R2": r2_score(y_test, y_pred_test),
            "Test_MAE": mean_absolute_error(y_test, y_pred_test),
            "Test_Spearman": spearmanr(y_test, y_pred_test).correlation,
            "rf": rf,
            "lasso": lasso
        }

    # --- grid search
    for ne in rf_param_grid["n_estimators"]:
        for md in rf_param_grid["max_depth"]:
            for msl in rf_param_grid["min_samples_leaf"]:
                for mss in rf_param_grid["min_samples_split"]:
                    for mf in rf_param_grid["max_features"]:
                        for alpha in lasso_alphas:
                            params = {
                                "n_estimators": ne,
                                "max_depth": md,
                                "min_samples_leaf": msl,
                                "min_samples_split": mss,
                                "max_features": mf
                            }
                            res = eval_combo(params, alpha)

                            print(
                                f"[HVG={n_genes}] RF={params}, "
                                f"Lasso α={alpha:.4f}, "
                                f"Test R²={res['Test_R2']:.3f}, "
                                f"Spearman={res['Test_Spearman']:.3f}"
                            )

                            if best is None or res["Test_Spearman"] > best["Test_Spearman"]:
                                best = {
                                    **res,
                                    "HVG": n_genes,
                                    "rf_params": params,
                                    "alpha": alpha
                                }

    # save detailed dict for this HVG
    detail_path = os.path.join(out_dir, f"best_LassoedForest_HVG{n_genes}.txt")
    with open(detail_path, "w") as f:
        f.write(str(best))

    print(f"\n🎯 Best for HVG={n_genes}:")
    print(best)
    print(f"📁 Saved → {detail_path}")

    # return a flat row for summary CSV
    return {
        "HVG": n_genes,
        "Train_R2": best["Train_R2"],
        "Test_R2": best["Test_R2"],
        "Test_MAE": best["Test_MAE"],
        "Test_Spearman": best["Test_Spearman"],
        "rf_n_estimators": best["rf_params"]["n_estimators"],
        "rf_max_depth": best["rf_params"]["max_depth"],
        "rf_min_samples_leaf": best["rf_params"]["min_samples_leaf"],
        "rf_min_samples_split": best["rf_params"]["min_samples_split"],
        "rf_max_features": best["rf_params"]["max_features"],
        "lasso_alpha": best["alpha"]
    }

# ------------------------------------------------------------
# Run tuning for each HVG value
# ------------------------------------------------------------
rows = []
for n in HVG_LIST:
    rows.append(tune_for_hvg(n))

summary_df = pd.DataFrame(rows)
summary_path = os.path.join(out_dir, "best_LassoedForest_by_HVG.csv")
summary_df.to_csv(summary_path, index=False)

print("\n✅ Finished multi-HVG tuning.")
print("📁 Summary →", summary_path)
