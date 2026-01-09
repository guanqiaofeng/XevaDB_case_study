#!/usr/bin/env python3
"""
16_run_HVG600_reproducible.py

Fully reproducible HVG=600 Lassoed Forest run.
Recreates EXACT tuning environment to obtain Test Spearman ≈ 0.75.
"""

import os
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr

# ---------------------------------------
# 1. Reproducibility: force all RNG seeds
# ---------------------------------------
np.random.seed(42)
random.seed(42)


# ---------------------------------------
# 2. Paths
# ---------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_800gene.csv"
out_dir = "../../data/ML_4model/HVG600_reproduce"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------
# 3. Load data
# ---------------------------------------
df = pd.read_csv(in_file)
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X_full = df.drop(columns=meta_cols)
y_full = np.log1p(df["final_DT_mean"].astype(float))

# ---------------------------------------
# 4. Use EXACT HVG600 gene list (matching tuning)
# ---------------------------------------
print("Selecting EXACT HVG=600 gene list (must match tuning)...")

var = X_full.var(axis=0)
HVG600_genes = var.sort_values(ascending=False).head(600).index.tolist()

# save HVG gene list for verification
pd.Series(HVG600_genes).to_csv(
    os.path.join(out_dir, "HVG600_gene_list_used.csv"),
    index=False
)

X = X_full[HVG600_genes]

print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} genes (HVG600)")

# ---------------------------------------
# 5. Standardize
# ---------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# 6. EXACT train/test split (same random_state=42)
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=0.2, random_state=42
)

print(f"Train={len(y_train)} | Test={len(y_test)}")

# ---------------------------------------
# 7. Tuned Parameters (from your grid search HVG=600)
# ---------------------------------------
rf_params = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}
lasso_alpha = 0.003727593720314938  # EXACT alpha


# ---------------------------------------
# 8. Train RandomForest
# ---------------------------------------
print("\nTraining Tuned Random Forest...")
rf = RandomForestRegressor(**rf_params)
rf.fit(X_train, y_train)

# leaf encoding
train_leaves = rf.apply(X_train)
test_leaves  = rf.apply(X_test)

# ---------------------------------------
# 9. Train Lasso on leaf encoding
# ---------------------------------------
print("Training Lasso on RF-leaf encoding...")

lasso = Lasso(alpha=lasso_alpha, max_iter=50000)
lasso.fit(train_leaves, y_train)

# predictions
y_pred_train = lasso.predict(train_leaves)
y_pred_test  = lasso.predict(test_leaves)

# ---------------------------------------
# 10. Metrics
# ---------------------------------------
Train_R2 = r2_score(y_train, y_pred_train)
Test_R2 = r2_score(y_test, y_pred_test)
Test_MAE = mean_absolute_error(y_test, y_pred_test)
Test_Spearman = spearmanr(y_test, y_pred_test).correlation

print("\n===== Reproducible HVG600 Lassoed Forest =====")
print(f"Train R²       : {Train_R2:.4f}")
print(f"Test R²        : {Test_R2:.4f}")
print(f"Test MAE       : {Test_MAE:.4f}")
print(f"Test Spearman  : {Test_Spearman:.4f}")   # should be ~0.75
print("==============================================")

# save results
pd.DataFrame([{
    "HVG": 600,
    "Train_R2": Train_R2,
    "Test_R2": Test_R2,
    "Test_MAE": Test_MAE,
    "Test_Spearman": Test_Spearman,
    **rf_params,
    "lasso_alpha": lasso_alpha
}]).to_csv(os.path.join(out_dir, "HVG600_results.csv"), index=False)


# ---------------------------------------
# 11. Plot: Observed vs Predicted
# ---------------------------------------
plt.figure(figsize=(4.5,4.5))
sns.regplot(
    x=y_test, y=y_pred_test,
    scatter_kws={'s':45, 'alpha':0.75},
    line_kws={'color':'steelblue'}
)
plt.xlabel("Observed log(DT+1)")
plt.ylabel("Predicted")
plt.title(f"Lassoed Forest HVG600\nTest Spearman={Test_Spearman:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "HVG600_scatter.png"), dpi=150)
plt.close()

print(f"\nPlots + results saved to: {out_dir}")
