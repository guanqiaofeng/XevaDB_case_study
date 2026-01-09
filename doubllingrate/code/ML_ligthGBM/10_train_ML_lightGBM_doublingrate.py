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

from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Base model
base_lgb = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    objective="regression",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "num_leaves": [4, 6, 8, 12, 16],
    "max_depth": [2, 3, 4, 5],
    "min_data_in_leaf": [5, 10, 15, 20],
    "feature_fraction": [0.5, 0.6, 0.7, 0.8],
    "bagging_fraction": [0.5, 0.6, 0.7, 0.8],
    "lambda_l1": [0.0, 0.1, 0.3, 0.5, 1.0],
    "lambda_l2": [0.0, 0.1, 0.5, 1.0, 2.0],
}

search = RandomizedSearchCV(
    estimator=base_lgb,
    param_distributions=param_grid,
    n_iter=40,              # 40 random trials
    scoring="neg_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="l2",
    callbacks=[
        early_stopping(stopping_rounds=100, verbose=False)
    ]
)

print("\nBest Params:", search.best_params_)

best_lgb = search.best_estimator_

# Evaluate final model
y_pred = best_lgb.predict(X_test)

print("\nFine-Tuned LightGBM:")
print("Test R²:", r2_score(y_test, y_pred))
print("Test MAE:", mean_absolute_error(y_test, y_pred))
print("Spearman:", spearmanr(y_test, y_pred).correlation)