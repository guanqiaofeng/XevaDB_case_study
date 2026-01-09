#!/usr/bin/env python3
"""
10_train_ML_doublingrate_v3.py

Train baseline ML regressors (ElasticNet, Ridge, Lasso)
to predict PDX doubling time from RNA-seq data.

New in this version:
 - Adds 5-fold cross-validation to estimate generalization (R² mean ± SD)
 - Adds optional log-transform of doubling time to stabilize variance
 - Keeps explicit 80/20 train-test split for fair evaluation
 - Uses equal sample weighting (no SD-based weighting)
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

# ------------------------------------------------------------
# --- Parameters ---
# ------------------------------------------------------------
USE_LOG_TARGET = True       # 🔹 Set True to use log(final_DT_mean)
TEST_SIZE = 0.2
TOP_N_FOLDS = 5

# ------------------------------------------------------------
# --- Paths ---
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_filtered.csv"
out_dir = "../../data/ML_v2/11"
os.makedirs(out_dir, exist_ok=True)

coef_out = os.path.join(out_dir, "ElasticNet_gene_coefficients.csv")
metrics_out = os.path.join(out_dir, "ML_model_performance_summary.csv")

# ------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X = df.drop(columns=meta_cols)
y_raw = df["final_DT_mean"]

# Optional log-transform of target
if USE_LOG_TARGET:
    y = np.log1p(y_raw)  # log(1 + y)
    target_label = "log(final_DT_mean + 1)"
    print("✅ Using log-transformed doubling time as target.")
else:
    y = y_raw.copy()
    target_label = "final_DT_mean"
    print("✅ Using raw doubling time as target.")

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
    X_scaled, y, test_size=TEST_SIZE, random_state=42
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ------------------------------------------------------------
# --- Helper for model evaluation ---
# ------------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Training performance
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    sp_train = spearmanr(y_train, y_pred_train).correlation

    # Test performance
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    sp_test = spearmanr(y_test, y_pred_test).correlation

    return {
        "r2_train": r2_train,
        "mae_train": mae_train,
        "sp_train": sp_train,
        "r2_test": r2_test,
        "mae_test": mae_test,
        "sp_test": sp_test,
        "y_pred_test": y_pred_test
    }

# ------------------------------------------------------------
# --- ElasticNetCV ---
# ------------------------------------------------------------
print("\nTraining ElasticNetCV...")
elastic = ElasticNetCV(
    l1_ratio=[.1, .3, .5, .7, .9, 1.0],
    cv=5,
    n_jobs=-1,
    random_state=42,
    max_iter=10000
)
res_el = evaluate_model(elastic, X_train, y_train, X_test, y_test)
coef = pd.Series(elastic.coef_, index=X.columns)
coef = coef[coef != 0].sort_values(ascending=False)
coef.to_csv(coef_out)

print(f"ElasticNetCV: Train R²={res_el['r2_train']:.3f}, "
      f"Test R²={res_el['r2_test']:.3f}, Test MAE={res_el['mae_test']:.2f}, "
      f"Test Spearman={res_el['sp_test']:.3f}")
print(f"Saved {len(coef)} non-zero gene coefficients → {coef_out}")

# --- 5-fold CV on full dataset (for robustness) ---
cv = KFold(n_splits=TOP_N_FOLDS, shuffle=True, random_state=42)
cv_scores = cross_val_score(elastic, X_scaled, y, cv=cv, scoring='r2')
print(f"5-fold CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ------------------------------------------------------------
# --- RidgeCV ---
# ------------------------------------------------------------
print("\nTraining RidgeCV...")
ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
sp_ridge = spearmanr(y_test, y_pred_ridge).correlation
print(f"RidgeCV: Test R²={r2_ridge:.3f}, MAE={mae_ridge:.2f}, Spearman={sp_ridge:.3f}")

# ------------------------------------------------------------
# --- LassoCV ---
# ------------------------------------------------------------
print("\nTraining LassoCV...")
lasso = LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
sp_lasso = spearmanr(y_test, y_pred_lasso).correlation
print(f"LassoCV: Test R²={r2_lasso:.3f}, MAE={mae_lasso:.2f}, Spearman={sp_lasso:.3f}")

# ------------------------------------------------------------
# --- Summarize results ---
# ------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["ElasticNetCV", "RidgeCV", "LassoCV"],
    "Train_R2": [res_el["r2_train"], np.nan, np.nan],
    "Test_R2": [res_el["r2_test"], r2_ridge, r2_lasso],
    "Test_MAE": [res_el["mae_test"], mae_ridge, mae_lasso],
    "Test_Spearman": [res_el["sp_test"], sp_ridge, sp_lasso],
    "n_features_nonzero": [len(coef), X.shape[1], np.sum(lasso.coef_ != 0)],
    "5fold_CV_R2_mean": [cv_scores.mean(), np.nan, np.nan],
    "5fold_CV_R2_sd": [cv_scores.std(), np.nan, np.nan],
    "Target": [target_label, target_label, target_label]
})
results.to_csv(metrics_out, index=False)
print(f"\n✅ Saved model performance summary → {metrics_out}")

# ------------------------------------------------------------
# --- Visualization ---
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5,4))
sns.regplot(x=y_test, y=res_el["y_pred_test"], scatter_kws={'s':40, 'alpha':0.7})
ax.set_xlabel(f"Observed {target_label} (Test)")
ax.set_ylabel("Predicted (ElasticNet)")
ax.set_title(f"Test Set: Observed vs Predicted (R²={res_el['r2_test']:.2f}, MAE={res_el['mae_test']:.1f})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ElasticNet_test_observed_vs_predicted.png"), dpi=150)
plt.close()

print("\n📊 Generated observed vs predicted plot (test set).")
print("🎯 ML training with cross-validation and log-transform option complete.")
