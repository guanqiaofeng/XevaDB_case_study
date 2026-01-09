#!/usr/bin/env python3
"""
15_compare_logtransform_ML.py

Compare model performance (ElasticNet, Random Forest, LightGBM)
using raw vs log-transformed doubling time as targets.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# ------------------------------------------------------------
# --- Parameters
# ------------------------------------------------------------
N_TOP_GENES = 800
N_SPLITS = 5
RANDOM_STATE = 42
OUT_DIR = "../../data/ML"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# --- Load data
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_filtered.csv"
df = pd.read_csv(in_file, dtype={"modelID": str})

meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]
X = df.drop(columns=meta_cols)
y_raw = df["final_DT_mean"].astype(float)
y_log = np.log(df["final_DT_mean"].astype(float))

# ------------------------------------------------------------
# --- Sample weights (inverse SD)
# ------------------------------------------------------------
sd = df["final_DT_sd"].replace(0, np.nan)
weights = 1 / (sd + 1e-3)
min_weight = np.nanpercentile(weights, 10)
weights = weights.fillna(min_weight)
weights = weights / weights.mean()

# ------------------------------------------------------------
# --- Feature selection (top N variance)
# ------------------------------------------------------------
gene_var = X.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(N_TOP_GENES).index
X_sel = X[top_genes]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# ------------------------------------------------------------
# --- Model definitions
# ------------------------------------------------------------
models = {
    "ElasticNet": ElasticNetCV(
        l1_ratio=[.1, .3, .5, .7, .9, 1.0],
        cv=N_SPLITS,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        max_iter=5000
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# ------------------------------------------------------------
# --- Evaluation function
# ------------------------------------------------------------
def evaluate_model(model, X, y, weights):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    r2s, maes, sps = [], [], []
    for train_idx, test_idx in cv.split(X):
        model.fit(X[train_idx], y.iloc[train_idx], sample_weight=weights.iloc[train_idx])
        y_pred = model.predict(X[test_idx])
        r2s.append(r2_score(y.iloc[test_idx], y_pred))
        maes.append(mean_absolute_error(y.iloc[test_idx], y_pred))
        sps.append(spearmanr(y.iloc[test_idx], y_pred).correlation)
    return np.mean(r2s), np.mean(maes), np.mean(sps)

# ------------------------------------------------------------
# --- Run experiments
# ------------------------------------------------------------
results = []

for target_name, y in {"Raw": y_raw, "Log": y_log}.items():
    print(f"\n🚀 Evaluating models with target = {target_name}")
    for model_name, model in models.items():
        r2, mae, sp = evaluate_model(model, X_scaled, y, weights)
        results.append({
            "Target": target_name,
            "Model": model_name,
            "R2_mean": r2,
            "MAE_mean": mae,
            "Spearman_mean": sp
        })
        print(f"{model_name} ({target_name}): R²={r2:.3f}, MAE={mae:.2f}, Spearman={sp:.3f}")

# ------------------------------------------------------------
# --- Save and plot summary
# ------------------------------------------------------------
res_df = pd.DataFrame(results)
res_out = os.path.join(OUT_DIR, "compare_logtransform_results.csv")
res_df.to_csv(res_out, index=False)
print(f"\n✅ Saved results → {res_out}")

plt.figure(figsize=(6,4))
sns.barplot(
    data=res_df,
    x="Model", y="R2_mean",
    hue="Target", palette="viridis"
)
plt.title("Cross-validated R² comparison (Raw vs Log target)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "compare_logtransform_R2.png"), dpi=150)
plt.close()

plt.figure(figsize=(6,4))
sns.barplot(
    data=res_df,
    x="Model", y="MAE_mean",
    hue="Target", palette="coolwarm"
)
plt.title("Cross-validated MAE comparison (Raw vs Log target)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "compare_logtransform_MAE.png"), dpi=150)
plt.close()

print("📊 Comparison plots saved.")
