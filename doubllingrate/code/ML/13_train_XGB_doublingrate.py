#!/usr/bin/env python3
"""
13_train_XGB_doublingrate.py

Train XGBoost and LightGBM regressors to predict log-transformed
PDX doubling time from RNA-seq expression data.

 - Uses inverse SD as sample weight
 - Selects top 500 high-variance genes
 - Reports 5-fold CV performance
 - Computes SHAP feature importance
"""

import os
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ------------------------------------------------------------
# --- Parameters
# ------------------------------------------------------------
N_TOP_GENES = 500
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
y = np.log1p(df["final_DT_mean"])  # log-transform
sd = df["final_DT_sd"].copy()

# ------------------------------------------------------------
# --- Sample weighting
# ------------------------------------------------------------
sd = sd.replace(0, np.nan)
weights = 1 / (sd + 1e-3)
min_weight = np.nanpercentile(weights, 10)
weights = weights.fillna(min_weight)
weights = weights / weights.mean()

print(f"Sample weights range: {weights.min():.3f} – {weights.max():.3f}")

# ------------------------------------------------------------
# --- Feature selection
# ------------------------------------------------------------
gene_var = X.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(N_TOP_GENES).index
X_sel = X[top_genes].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

print(f"Selected top {N_TOP_GENES} high-variance genes for modeling.")

# ------------------------------------------------------------
# --- Model configs
# ------------------------------------------------------------
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

models = {"XGBoost": xgb_model, "LightGBM": lgbm_model}

# ------------------------------------------------------------
# --- Cross-validation
# ------------------------------------------------------------
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
perf_records = []

for model_name, model in models.items():
    print(f"\n🚀 Training {model_name}...")
    r2_list, mae_list, sp_list = [], [], []
    y_pred_all = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled)):
        model.fit(X_scaled[train_idx], y[train_idx], sample_weight=weights.iloc[train_idx])
        y_pred = model.predict(X_scaled[test_idx])
        y_pred_all[test_idx] = y_pred

        r2 = r2_score(y[test_idx], y_pred)
        mae = mean_absolute_error(y[test_idx], y_pred)
        sp = spearmanr(y[test_idx], y_pred).correlation
        r2_list.append(r2)
        mae_list.append(mae)
        sp_list.append(sp)
        print(f"Fold {fold+1}: R²={r2:.3f}, MAE={mae:.2f}, Spearman={sp:.3f}")

    # Save performance
    perf_records.append({
        "Model": model_name,
        "R2_mean": np.mean(r2_list),
        "R2_sd": np.std(r2_list),
        "MAE_mean": np.mean(mae_list),
        "Spearman_mean": np.mean(sp_list)
    })

    # Plot observed vs predicted
    fig, ax = plt.subplots(figsize=(4.5,4))
    sns.regplot(x=y, y=y_pred_all, scatter_kws={'s':40, 'alpha':0.7})
    ax.set_xlabel("Observed log(Doubling Time)")
    ax.set_ylabel(f"Predicted ({model_name})")
    ax.set_title(f"{model_name} (R²={np.mean(r2_list):.2f}, MAE={np.mean(mae_list):.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Observed_vs_Predicted_{model_name}.png"), dpi=150)
    plt.close()

    # Refit full model for SHAP
    model.fit(X_scaled, y, sample_weight=weights)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    # Compute mean(|SHAP|) per gene
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({
        "Gene": top_genes,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False)

    shap_df.to_csv(os.path.join(OUT_DIR, f"{model_name}_SHAP_topgenes.csv"), index=False)
    print(f"✅ Saved top SHAP genes → {model_name}_SHAP_topgenes.csv")

    # Plot top 20 SHAP genes
    plt.figure(figsize=(6,4))
    sns.barplot(x="MeanAbsSHAP", y="Gene", data=shap_df.head(20), palette="viridis")
    plt.title(f"Top 20 SHAP genes ({model_name})")
    plt.xlabel("Mean |SHAP value| (feature importance)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{model_name}_SHAP_top20.png"), dpi=150)
    plt.close()

# Save performance summary
perf_df = pd.DataFrame(perf_records)
perf_df.to_csv(os.path.join(OUT_DIR, "XGB_LGBM_performance_summary.csv"), index=False)
print(f"\n✅ Saved performance summary → XGB_LGBM_performance_summary.csv")
print("🎯 Gradient boosting training and SHAP interpretation complete.")
