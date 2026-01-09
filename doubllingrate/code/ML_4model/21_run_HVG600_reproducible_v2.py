#!/usr/bin/env python3
"""
21_run_HVG600_reproducible_v2.py

Reproduce HVG=600 results for:
  - ElasticNet
  - LightGBM
  - RandomForest (tuned baseline)
  - Lassoed Forest (tuned for HVG=600)

Uses:
  - Input: RNAseq_with_doublingrate_800gene.csv
  - Target: log(final_DT_mean + 1)
  - HVG selection: top 600 most variable genes
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import ElasticNetCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

sns.set(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams.update({"figure.dpi": 150})

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_800gene.csv"
out_dir = "../../data/ML_4model/HVG600_reproduce_v2"
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
# HVG = 600 selection (must match tuner)
# ------------------------------------------------------------
print("Selecting EXACT HVG=600 gene list (top variance)...")
var = X_full.var(axis=0)
top600 = var.sort_values(ascending=False).head(600).index.tolist()

X = X_full[top600].copy()
y = y_full.copy()

print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} genes (HVG600)")

# ------------------------------------------------------------
# Scale + split
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train = {X_train.shape[0]} | Test = {X_test.shape[0]}")

def eval_metrics(y_train, y_test, y_pred_train, y_pred_test):
    return {
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2":  r2_score(y_test,  y_pred_test),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Test_Spearman": spearmanr(y_test, y_pred_test).correlation
    }

def scatter_plot(y_true, y_pred, title, fname):
    plt.figure(figsize=(4.2, 4))
    sns.regplot(
        x=y_true, y=y_pred,
        scatter_kws={'s':45, 'alpha':0.75},
        line_kws={'color': 'steelblue'}
    )
    plt.xlabel("Observed log(DT+1)")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()

rows = []

# ------------------------------------------------------------
# 1) ElasticNet
# ------------------------------------------------------------
print("\n=== ElasticNet (HVG600) ===")
enet = ElasticNetCV(
    alphas=np.logspace(-4, 2, 60),
    l1_ratio=np.linspace(0.1, 1.0, 10),
    cv=5,
    n_jobs=-1,
    max_iter=50000,
    random_state=42
)
enet.fit(X_train, y_train)

enet_train = enet.predict(X_train)
enet_test  = enet.predict(X_test)

enet_metrics = eval_metrics(y_train, y_test, enet_train, enet_test)
enet_metrics["Model"] = "ElasticNet"
rows.append({"Model": "ElasticNet", **enet_metrics})

print(enet_metrics)
scatter_plot(y_test, enet_test, "ElasticNet (HVG600)", "EN_scatter_HVG600.png")

# ------------------------------------------------------------
# 2) LightGBM
# ------------------------------------------------------------
print("\n=== LightGBM (HVG600) ===")
lgb = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=3,
    num_leaves=8,
    min_child_samples=10,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l1=0.5,
    lambda_l2=1.0,
    objective="regression",
    random_state=42,
    n_jobs=-1
)

lgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="l2",
    callbacks=[early_stopping(stopping_rounds=100),
               log_evaluation(period=50)]
)

lgb_train = lgb.predict(X_train)
lgb_test  = lgb.predict(X_test)

lgb_metrics = eval_metrics(y_train, y_test, lgb_train, lgb_test)
lgb_metrics["Model"] = "LightGBM"
rows.append({"Model": "LightGBM", **lgb_metrics})

print(lgb_metrics)
scatter_plot(y_test, lgb_test, "LightGBM (HVG600)", "LGBM_scatter_HVG600.png")

# ------------------------------------------------------------
# 3) Tuned RandomForest (baseline RF)
#    You can choose to align this with LassoedForest RF params (HVG=600)
# ------------------------------------------------------------
print("\n=== RandomForest (HVG600) ===")
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
rf_train = rf.predict(X_train)
rf_test  = rf.predict(X_test)

rf_metrics = eval_metrics(y_train, y_test, rf_train, rf_test)
rf_metrics["Model"] = "RandomForest"
rows.append({"Model": "RandomForest", **rf_metrics})

print(rf_metrics)
scatter_plot(y_test, rf_test, "RandomForest (HVG600)", "RF_scatter_HVG600.png")

# ------------------------------------------------------------
# 4) Tuned Lassoed Forest (HVG600 parameters from tuning)
# ------------------------------------------------------------
print("\n=== Lassoed Forest (HVG600 tuned) ===")

tuned_rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=4,
    min_samples_leaf=2,
    min_samples_split=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
tuned_rf.fit(X_train, y_train)

# RF leaf encoding
leaves_train = tuned_rf.apply(X_train)
leaves_test  = tuned_rf.apply(X_test)

tuned_lasso = Lasso(alpha=0.0517947467923121, max_iter=50000)
tuned_lasso.fit(leaves_train, y_train)

lf_train = tuned_lasso.predict(leaves_train)
lf_test  = tuned_lasso.predict(leaves_test)

lf_metrics = eval_metrics(y_train, y_test, lf_train, lf_test)
lf_metrics["Model"] = "LassoedForest"
rows.append({"Model": "LassoedForest", **lf_metrics})

print("\n===== Reproducible HVG600 Lassoed Forest (800gene) =====")
print(lf_metrics)
print("========================================================")

scatter_plot(y_test, lf_test, "LassoedForest (HVG600)", "LassoForest_scatter_HVG600.png")

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------
summary_df = pd.DataFrame(rows)
summary_path = os.path.join(out_dir, "HVG600_model_summary.csv")
summary_df.to_csv(summary_path, index=False, float_format="%.4f")

print("\n📁 Saved summary →", summary_path)
print("All plots + outputs saved in:", out_dir)
