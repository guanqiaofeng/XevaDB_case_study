#!/usr/bin/env python3
"""
20_run_HVG600_models.py

Run 4 ML models on HVG = 600 gene set:
 - ElasticNetCV
 - LightGBM
 - RandomForest
 - Lassoed Forest (TUNED from HVG600 search)

Target: log(final_DT_mean + 1)

Outputs:
 - Model summary CSV
 - Scatter plots
 - Comparison barplot (Test Spearman)
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

# Models
from sklearn.linear_model import ElasticNetCV, Lasso
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_800gene.csv"
out_dir = "../../data/ML_4model/20_HVG600"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X_full = df.drop(columns=meta_cols)
y_full = np.log1p(df["final_DT_mean"].astype(float))

print(f"Loaded: {X_full.shape[0]} samples × {X_full.shape[1]} genes")

# ------------------------------------------------------------
# Select HVG = 600 genes
# ------------------------------------------------------------
var = X_full.var(axis=0)
top600 = var.sort_values(ascending=False).head(600).index.tolist()
X = X_full[top600]

# ------------------------------------------------------------
# Scale
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=0.2, random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------------------------------
# Helper: evaluate metrics
# ------------------------------------------------------------
def evaluate(y_train, y_test, y_pred_train, y_pred_test):
    return {
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Test_Spearman": spearmanr(y_test, y_pred_test).correlation
    }

# ------------------------------------------------------------
# Helper: scatter plot
# ------------------------------------------------------------
def plot_scatter(y_true, y_pred, title, fname):
    plt.figure(figsize=(4.5,4))
    sns.regplot(
        x=y_true, y=y_pred,
        scatter_kws={'s':50, 'alpha':0.8},
        line_kws={'color':'steelblue'}
    )
    plt.xlabel("Observed log(DT+1)")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ------------------------------------------------------------
# === ElasticNet ===
# ------------------------------------------------------------
print("\nElasticNet...")
enet = ElasticNetCV(
    alphas=np.logspace(-4, 2, 60),
    l1_ratio=np.linspace(0.1, 1.0, 10),
    cv=5, n_jobs=-1, max_iter=50000, random_state=42
)
enet.fit(X_train, y_train)

enet_train = enet.predict(X_train)
enet_test  = enet.predict(X_test)

enet_metrics = evaluate(y_train, y_test, enet_train, enet_test)

plot_scatter(
    y_test, enet_test,
    "ElasticNet (HVG = 600)",
    os.path.join(out_dir, "EN_scatter.png")
)

# ------------------------------------------------------------
# === LightGBM ===
# ------------------------------------------------------------
print("\nLightGBM...")
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
    random_state=42
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

lgb_metrics = evaluate(y_train, y_test, lgb_train, lgb_test)

plot_scatter(
    y_test, lgb_test,
    "LightGBM (HVG = 600)",
    os.path.join(out_dir, "LGBM_scatter.png")
)

# ------------------------------------------------------------
# === RandomForest ===
# (use tuned baseline RF parameters)
# ------------------------------------------------------------
print("\nRandomForest...")
rf = RandomForestRegressor(
    n_estimators=1000,
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

rf_metrics = evaluate(y_train, y_test, rf_train, rf_test)

plot_scatter(
    y_test, rf_test,
    "RandomForest (HVG = 600)",
    os.path.join(out_dir, "RF_scatter.png")
)

# ------------------------------------------------------------
# === Lassoed Forest (TUNED from HVG=600 search)
# ------------------------------------------------------------
print("\nLassoed Forest (TUNED)...")

lf_rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=2,
    min_samples_split=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
lf_rf.fit(X_train, y_train)

# Leaves → encoding
leaves_train = lf_rf.apply(X_train)
leaves_test  = lf_rf.apply(X_test)

lf_lasso = Lasso(alpha=0.0037, max_iter=50000)
lf_lasso.fit(leaves_train, y_train)

lf_train = lf_lasso.predict(leaves_train)
lf_test  = lf_lasso.predict(leaves_test)

lf_metrics = evaluate(y_train, y_test, lf_train, lf_test)

plot_scatter(
    y_test, lf_test,
    "Lassoed Forest (HVG = 600, Tuned)",
    os.path.join(out_dir, "LassoForest_scatter.png")
)

# ------------------------------------------------------------
# === Save summary
# ------------------------------------------------------------
summary = pd.DataFrame([
    {"Model": "ElasticNet", **enet_metrics},
    {"Model": "LightGBM", **lgb_metrics},
    {"Model": "RandomForest", **rf_metrics},
    {"Model": "LassoedForest", **lf_metrics},
])

summary.to_csv(os.path.join(out_dir, "HVG600_model_summary.csv"), index=False)

print("\n📁 Saved summary →", os.path.join(out_dir, "HVG600_model_summary.csv"))

# ------------------------------------------------------------
# === Comparison barplot (Spearman)
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.barplot(
    data=summary,
    x="Model",
    y="Test_Spearman",
    palette="viridis"
)
plt.ylim(0, summary["Test_Spearman"].max() * 1.15)
plt.title("Test Spearman Comparison (HVG = 600)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "HVG600_Spearman_barplot.png"), dpi=150)
plt.close()

print("🎯 Done. All outputs saved in:", out_dir)
