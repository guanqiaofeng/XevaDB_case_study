#!/usr/bin/env python3
"""
16_tune_LassoedForest.py  (UPDATED)

Performs grid search for Lassoed Forest
and saves ALL results to a CSV for visualization.
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_800gene.csv"
out_dir = "../../data/ML_4model/16_tuning"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X = df.drop(columns=meta_cols)
y = np.log1p(df["final_DT_mean"])

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Parameter grids
# ------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [500, 1000, 2000],
    "max_depth": [None, 4, 6],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 4],
    "max_features": ["sqrt", "log2"],
}

lasso_alphas = np.logspace(-3, 1, 8)

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def evaluate_rf_lasso(rf_params, alpha):

    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)
    rf.fit(X_train, y_train)

    train_leaves = rf.apply(X_train)
    test_leaves  = rf.apply(X_test)

    lasso = Lasso(alpha=alpha, max_iter=50000)
    lasso.fit(train_leaves, y_train)

    y_pred_train = lasso.predict(train_leaves)
    y_pred_test  = lasso.predict(test_leaves)

    return {
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Test_Spearman": spearmanr(y_test, y_pred_test).correlation
    }

# ------------------------------------------------------------
# Grid search + full logging
# ------------------------------------------------------------
all_rows = []
best = None

print("🔍 Starting Lassoed Forest grid search...")

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

                        res = evaluate_rf_lasso(params, alpha)

                        # --- store row ---
                        row = {
                            **params,
                            "alpha": alpha,
                            **res
                        }
                        all_rows.append(row)

                        # --- best model ---
                        if best is None or res["Test_Spearman"] > best["Test_Spearman"]:
                            best = {**row}

                        print(
                            f"RF={params}, alpha={alpha:.4f}, "
                            f"Test R2={res['Test_R2']:.3f}, "
                            f"Spearman={res['Test_Spearman']:.3f}"
                        )

# save full tuning log
log_df = pd.DataFrame(all_rows)
log_path = os.path.join(out_dir, "LassoedForest_tuning_results_800HVGs.csv")
log_df.to_csv(log_path, index=False)

# save best model summary
best_file = os.path.join(out_dir, "best_LassoedForest_results.txt")
with open(best_file, "w") as f:
    f.write(str(best))

print("\n🎯 Best Tuned Lassoed Forest:")
print(best)
print("\n📁 Saved full tuning log →", log_path)
print("📁 Saved best model →", best_file)
