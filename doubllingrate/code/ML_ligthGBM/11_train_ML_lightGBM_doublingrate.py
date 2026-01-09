#!/usr/bin/env python3
"""
10_train_ML_lightGBM_doublingrate.py

Train LightGBM with strong regularization
to predict PDX doubling time from RNA-seq data.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

from lightgbm import LGBMRegressor

# ------------------------------------------------------------
# --- Paths ---
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_797gene.csv"
out_dir = "../../data/ML_lightGBM"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X = df.drop(columns=meta_cols)
y = df["final_DT_mean"]

print(f"Data shape: {X.shape[0]} samples × {X.shape[1]} genes")

# ------------------------------------------------------------
# --- Standardize features ---
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# --- Train-test split ---
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ------------------------------------------------------------
# --- LightGBM Model ---
# ------------------------------------------------------------
print("\nTraining LightGBM (strong regularization)...")

lgb_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=3,
    num_leaves=8,
    min_data_in_leaf=10,
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

from lightgbm import early_stopping, log_evaluation

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="l2",
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=50)
    ]
)

# ------------------------------------------------------------
# --- Evaluate ---
# ------------------------------------------------------------
y_pred_train = lgb_model.predict(X_train)
y_pred_test = lgb_model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
sp_train = spearmanr(y_train, y_pred_train).correlation

r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
sp_test = spearmanr(y_test, y_pred_test).correlation

print(f"LightGBM Results:")
print(f"  Train R²:      {r2_train:.3f}")
print(f"  Test  R²:      {r2_test:.3f}")
print(f"  Test  MAE:     {mae_test:.2f}")
print(f"  Test  Spearman:{sp_test:.3f}")

# ------------------------------------------------------------
# --- Save scatter plot ---
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5,4))
sns.regplot(x=y_test, y=y_pred_test, scatter_kws={'s':40, 'alpha':0.7})
ax.set_xlabel("Observed DT (Test)")
ax.set_ylabel("Predicted DT (LightGBM)")
ax.set_title(f"LightGBM: R²={r2_test:.2f}, MAE={mae_test:.1f}")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "LightGBM_observed_vs_predicted.png"), dpi=150)
plt.close()

print("📊 Saved LightGBM observed vs predicted plot.")
print("🎯 LightGBM training complete.")
