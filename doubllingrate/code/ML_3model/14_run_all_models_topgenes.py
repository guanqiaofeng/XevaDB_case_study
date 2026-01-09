#!/usr/bin/env python3
"""
14_run_all_models_topgenes.py

Run ElasticNet, LightGBM, CatBoost using:
 - full gene set, OR
 - top-N highly variable genes (via --top_genes N)

Matches pipelines of your 13_* production scripts.
Outputs:
 - unified summary CSV
 - observed vs predicted plots for each model
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

# Models
from sklearn.linear_model import ElasticNetCV
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor, Pool


# ------------------------------------------------------------
# --- CLI
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run 3 models with optional HVG filtering.")
parser.add_argument("--top_genes", type=int, default=None,
                    help="Number of top-variance genes to keep (e.g., 300). Default = all genes.")
args = parser.parse_args()

# ------------------------------------------------------------
# --- Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_797gene.csv"
out_dir = "../../data/ML_v2/14_topgenes"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X = df.drop(columns=meta_cols)
y = np.log1p(df["final_DT_mean"].astype(float))

print(f"📌 Loaded {X.shape[0]} samples × {X.shape[1]} genes")

# ------------------------------------------------------------
# --- Top-variance gene selection
# ------------------------------------------------------------
if args.top_genes is not None:
    var = X.var(axis=0)
    top_genes = var.sort_values(ascending=False).head(args.top_genes).index.tolist()
    X = X[top_genes]
    print(f"🔬 Using top {args.top_genes} most variable genes → now {X.shape[1]} genes")
else:
    print("🔬 Using ALL genes")

# ------------------------------------------------------------
# --- Standardize
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# --- Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

train_n, val_n = X_train.shape[0], X_test.shape[0]
print(f"Train = {train_n} | Test = {val_n}")

# ------------------------------------------------------------
# --- Helper: compute metrics
# ------------------------------------------------------------
def evaluate(y_train, y_test, y_pred_train, y_pred_test):
    return {
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Test_Spearman": spearmanr(y_test, y_pred_test).correlation
    }

# ------------------------------------------------------------
# === ElasticNet ===
# ------------------------------------------------------------
print("\n=== ElasticNet ===")

enet = ElasticNetCV(
    alphas=np.logspace(-4, 2, 60),
    l1_ratio=np.linspace(0.1, 1.0, 10),
    cv=5, n_jobs=-1, max_iter=50000, random_state=42
)
enet.fit(X_train, y_train)

enet_train = enet.predict(X_train)
enet_test  = enet.predict(X_test)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(enet, X_scaled, y, cv=cv, scoring="r2")

enet_metrics = evaluate(y_train, y_test, enet_train, enet_test)
enet_metrics["CV_R2_mean"] = cv_scores.mean()
enet_metrics["CV_R2_SD"] = cv_scores.std()
print(enet_metrics)

# ------------------------------------------------------------
# === LightGBM ===
# ------------------------------------------------------------
print("\n=== LightGBM ===")

lgb = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=3,
    num_leaves=8,
    min_child_samples=10,
    min_data_in_leaf=10,
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
lgb_test = lgb.predict(X_test)

cv_scores = cross_val_score(lgb, X_scaled, y, cv=cv, scoring="r2")

lgb_metrics = evaluate(y_train, y_test, lgb_train, lgb_test)
lgb_metrics["CV_R2_mean"] = cv_scores.mean()
lgb_metrics["CV_R2_SD"] = cv_scores.std()
lgb_metrics["Best_iteration"] = lgb.best_iteration_

print(lgb_metrics)

# ------------------------------------------------------------
# === CatBoost ===
# ------------------------------------------------------------
print("\n=== CatBoost ===")

train_pool = Pool(X_train, y_train)
test_pool  = Pool(X_test, y_test)

cat = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    loss_function="RMSE",
    od_type="Iter", od_wait=100,
    random_seed=42,
    verbose=False
)
cat.fit(train_pool, eval_set=test_pool)

cat_train = cat.predict(X_train)
cat_test  = cat.predict(X_test)

cv_scores = cross_val_score(cat, X_scaled, y, cv=cv, scoring="r2")

cat_metrics = evaluate(y_train, y_test, cat_train, cat_test)
cat_metrics["CV_R2_mean"] = cv_scores.mean()
cat_metrics["CV_R2_SD"] = cv_scores.std()
cat_metrics["Best_iteration"] = cat.get_best_iteration()

print(cat_metrics)

# ------------------------------------------------------------
# --- Save unified summary
# ------------------------------------------------------------
summary = pd.DataFrame([
    {"Model": "ElasticNet", **enet_metrics},
    {"Model": "LightGBM",  **lgb_metrics},
    {"Model": "CatBoost",  **cat_metrics},
])

suffix = f"_top{args.top_genes}" if args.top_genes else "_allgenes"
summary_out = os.path.join(out_dir, f"model_summary{suffix}.csv")
summary.to_csv(summary_out, index=False)

print(f"\n📁 Saved summary → {summary_out}")

# ------------------------------------------------------------
# --- Scatter plots
# ------------------------------------------------------------
def plot_scatter(y_true, y_pred, title, fname):
    plt.figure(figsize=(4.5,4))
    sns.regplot(x=y_true, y=y_pred,
                scatter_kws={'s':50, 'alpha':0.8},
                line_kws={'color':'steelblue'})
    plt.xlabel("Observed log(DT+1)")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()

plot_scatter(y_test, enet_test, "ElasticNet", f"EN_scatter{suffix}.png")
plot_scatter(y_test, lgb_test, "LightGBM", f"LGBM_scatter{suffix}.png")
plot_scatter(y_test, cat_test, "CatBoost", f"CatBoost_scatter{suffix}.png")

print("\n🎯 Done. All outputs saved to:", out_dir)
