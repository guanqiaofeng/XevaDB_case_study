#!/usr/bin/env python3
"""
15_run_multi_HVG_experiment.py  (UPDATED with tuned Lassoed Forest)

Runs ElasticNet, LightGBM, RandomForest, and TUNED Lassoed Forest
across different HVG thresholds (top variance genes).

Outputs:
 - unified CSV summary
 - scatter plots
 - heatmaps of model performance
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import ElasticNetCV, Lasso
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------
# --- HVG List
# ------------------------------------------------------------
HVG_LIST = [300, 400, 500, 600, 700, 800]

# ------------------------------------------------------------
# --- Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_797gene.csv"
out_dir = "../../data/ML_4model/15_multiHVG_tuned600_797"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load base data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X_full = df.drop(columns=meta_cols)
y_full = np.log1p(df["final_DT_mean"].astype(float))

print(f"Loaded: {X_full.shape[0]} samples × {X_full.shape[1]} genes")

# ------------------------------------------------------------
# --- Helper functions
# ------------------------------------------------------------
def evaluate_metrics(y_train, y_test, y_pred_train, y_pred_test):
    return {
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Test_Spearman": spearmanr(y_test, y_pred_test).correlation
    }

def plot_scatter(y_true, y_pred, title, fname):
    plt.figure(figsize=(4.2, 4))
    sns.regplot(
        x=y_true, y=y_pred,
        scatter_kws={'s':45, 'alpha':0.75},
        line_kws={'color':'steelblue'}
    )
    plt.xlabel("Observed log(DT+1)")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ------------------------------------------------------------
# --- Storage
# ------------------------------------------------------------
rows = []

# ------------------------------------------------------------
# --- Loop over HVG settings
# ------------------------------------------------------------
for n_genes in HVG_LIST:
    print("\n" + "="*70)
    print(f"🔬 Running models for top {n_genes} HVGs")
    print("="*70)

    # Select HVGs
    var = X_full.var(axis=0)
    top_genes = var.sort_values(ascending=False).head(n_genes).index.tolist()
    X = X_full[top_genes]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_full, test_size=0.2, random_state=42
    )

    # KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # === ElasticNet ===
    # ------------------------------------------------------------------
    print("\nElasticNet...")
    enet = ElasticNetCV(
        alphas=np.logspace(-4, 2, 60),
        l1_ratio=np.linspace(0.1, 1.0, 10),
        cv=5, n_jobs=-1, max_iter=50000, random_state=42
    )
    enet.fit(X_train, y_train)
    en_train_pred = enet.predict(X_train)
    en_test_pred  = enet.predict(X_test)

    en_metrics = evaluate_metrics(y_train, y_test, en_train_pred, en_test_pred)
    cv_scores = cross_val_score(enet, X_scaled, y_full, cv=cv, scoring="r2")
    en_metrics["CV_R2_mean"] = cv_scores.mean()
    en_metrics["CV_R2_SD"] = cv_scores.std()

    rows.append({"Model": "ElasticNet", "n_genes": n_genes, **en_metrics})

    plot_scatter(
        y_test, en_test_pred,
        f"ElasticNet (top {n_genes})",
        os.path.join(out_dir, f"EN_scatter_top{n_genes}.png")
    )

    # ------------------------------------------------------------------
    # === LightGBM ===
    # ------------------------------------------------------------------
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

    lgb_train_pred = lgb.predict(X_train)
    lgb_test_pred  = lgb.predict(X_test)

    lgb_metrics = evaluate_metrics(y_train, y_test, lgb_train_pred, lgb_test_pred)
    cv_scores = cross_val_score(lgb, X_scaled, y_full, cv=cv, scoring="r2")
    lgb_metrics["CV_R2_mean"] = cv_scores.mean()
    lgb_metrics["CV_R2_SD"] = cv_scores.std()

    rows.append({"Model": "LightGBM", "n_genes": n_genes, **lgb_metrics})

    plot_scatter(
        y_test, lgb_test_pred,
        f"LightGBM (top {n_genes})",
        os.path.join(out_dir, f"LGBM_scatter_top{n_genes}.png")
    )

    # ------------------------------------------------------------------
    # === Tuned Random Forest (baseline RF) ===
    # ------------------------------------------------------------------
    print("\nRandomForest (baseline)...")
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
    rf_train_pred = rf.predict(X_train)
    rf_test_pred = rf.predict(X_test)

    rf_metrics = evaluate_metrics(y_train, y_test, rf_train_pred, rf_test_pred)
    cv_scores = cross_val_score(rf, X_scaled, y_full, cv=cv, scoring="r2")
    rf_metrics["CV_R2_mean"] = cv_scores.mean()
    rf_metrics["CV_R2_SD"] = cv_scores.std()

    rows.append({"Model": "RandomForest", "n_genes": n_genes, **rf_metrics})

    plot_scatter(
        y_test, rf_test_pred,
        f"RandomForest (top {n_genes})",
        os.path.join(out_dir, f"RF_scatter_top{n_genes}.png")
    )

    # ------------------------------------------------------------------
    # === Tuned Lassoed Forest ===
    # ------------------------------------------------------------------
    print("\nLassoed Forest (TUNED)...")

    # tuned RF params from HVG=600 search
    tuned_rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    tuned_rf.fit(X_train, y_train)

    # Leaves encoding
    leaves_train = tuned_rf.apply(X_train)
    leaves_test  = tuned_rf.apply(X_test)

    tuned_lasso = Lasso(alpha=0.003727593720314938, max_iter=50000)
    tuned_lasso.fit(leaves_train, y_train)

    lf_train_pred = tuned_lasso.predict(leaves_train)
    lf_test_pred  = tuned_lasso.predict(leaves_test)

    lf_metrics = evaluate_metrics(y_train, y_test, lf_train_pred, lf_test_pred)

    rows.append({"Model": "LassoedForest", "n_genes": n_genes, **lf_metrics})

    plot_scatter(
        y_test, lf_test_pred,
        f"LassoedForest (top {n_genes})",
        os.path.join(out_dir, f"LassoForest_scatter_top{n_genes}.png")
    )

# ------------------------------------------------------------
# --- Save summary
# ------------------------------------------------------------
summary_df = pd.DataFrame(rows)
summary_out = os.path.join(out_dir, "multiHVG_summary_tuned.csv")
summary_df.to_csv(summary_out, index=False)

print("\n📁 Saved unified summary →", summary_out)

# ------------------------------------------------------------
# --- Heatmap
# ------------------------------------------------------------
pivot = summary_df.pivot(index="Model", columns="n_genes", values="Test_R2")

plt.figure(figsize=(9, 4))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
plt.title("Test R² across HVG settings (with Tuned Lassoed Forest)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "heatmap_TestR2.png"), dpi=150)
plt.close()

print("🎯 Done. Outputs saved to:", out_dir)
