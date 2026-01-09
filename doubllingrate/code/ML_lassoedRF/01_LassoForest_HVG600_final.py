#!/usr/bin/env python3
"""
01_LassoForest_HVG600_final.py

Final production-grade HVG=600 Lassoed Forest model.

Outputs:
 - rf_model.joblib
 - lasso_model.joblib
 - scaler.joblib
 - HVG600_genes.txt
 - LassoForest_leaf_features.csv
 - LassoForest_lasso_coefficients.csv
 - LassoForest_model_summary.json
 - LassoForest_scatter.png
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

sns.set(style="whitegrid", context="paper", font_scale=0.95)
plt.rcParams.update({"figure.dpi": 150})

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_800gene.csv"
out_dir = "../../data/ML_lassoedRF/HVG600_LassoForest"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X_full = df.drop(columns=meta_cols)
y_full = np.log1p(df["final_DT_mean"].astype(float))

print(f"Loaded matrix: {X_full.shape[0]} samples × {X_full.shape[1]} genes")

# ------------------------------------------------------------
# Select exact HVG600 genes (top variance)
# ------------------------------------------------------------
print("Selecting HVG600...")
var = X_full.var(axis=0)
top600 = var.sort_values(ascending=False).head(600).index.tolist()

X = X_full[top600].copy()
y = y_full.copy()

with open(os.path.join(out_dir, "HVG600_genes.txt"), "w") as f:
    for g in top600:
        f.write(g + "\n")

print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} genes")

# ------------------------------------------------------------
# Scale + split
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Final tuned Lassoed Forest
# ------------------------------------------------------------
print("\n=== TRAINING FINAL LASSOED FOREST (HVG600) ===")

# Tuned RF parameters (from HVG600 tuning table)
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=4,
    min_samples_leaf=2,
    min_samples_split=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(out_dir, "rf_model.joblib"))

# Leaf encoding
train_leaves = rf.apply(X_train)
test_leaves  = rf.apply(X_test)

# Tuned Lasso alpha for HVG600
lasso = Lasso(alpha=0.0517947467923121, max_iter=50000)
lasso.fit(train_leaves, y_train)
joblib.dump(lasso, os.path.join(out_dir, "lasso_model.joblib"))

# Predictions
train_pred = lasso.predict(train_leaves)
test_pred  = lasso.predict(test_leaves)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
metrics = {
    "Train_R2": float(r2_score(y_train, train_pred)),
    "Test_R2":  float(r2_score(y_test, test_pred)),
    "Test_MAE": float(mean_absolute_error(y_test, test_pred)),
    "Test_Spearman": float(spearmanr(y_test, test_pred).correlation),
    "n_genes": 600,
    "rf_params": {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "max_features": "sqrt"
    },
    "lasso_alpha": 0.0517947467923121
}

with open(os.path.join(out_dir, "LassoForest_model_summary.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\n===== FINAL HVG600 LASSOED FOREST MODEL =====")
print(json.dumps(metrics, indent=4))
print("============================================\n")

# ------------------------------------------------------------
# Save Lasso coefficients (leaf-feature importance)
# ------------------------------------------------------------
coef = pd.Series(lasso.coef_, index=[f"leaf_{i}" for i in range(train_leaves.shape[1])])
coef.to_csv(os.path.join(out_dir, "LassoForest_lasso_coefficients.csv"))

# Save the raw leaf design matrix (optional)
np.savetxt(os.path.join(out_dir, "LassoForest_leaf_features.csv"),
           train_leaves, delimiter=",")

# ------------------------------------------------------------
# Plot scatter
# ------------------------------------------------------------
plt.figure(figsize=(4.5,4))
sns.regplot(
    x=y_test, y=test_pred,
    scatter_kws={'s':50, 'alpha':0.75},
    line_kws={'color': 'steelblue'}
)
plt.xlabel("Observed log(DT+1)")
plt.ylabel("Predicted")
plt.title("Lassoed Forest (HVG600)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "LassoForest_scatter.png"), dpi=150)
plt.close()

print("🎯 All final model outputs saved to:")
print(out_dir)
