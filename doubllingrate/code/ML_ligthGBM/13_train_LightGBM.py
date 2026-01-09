#!/usr/bin/env python3
"""
13_train_LightGBM.py

LightGBM model using log-transformed doubling time (ln(DT + 1)) target.
Mirrors the preprocessing and evaluation pipeline of 13_train_ElasticNet.py.

Includes:
- Strong regularization
- Early stopping (callback-based)
- SHAP interpretation (if installed)
- Train/test split, CV R², Spearman, MAE, plots
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Optional SHAP
try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

# ------------------------------------------------------------
# --- Setup
# ------------------------------------------------------------
sns.set(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams.update({"figure.dpi": 150})

in_file = "../../data/analyze/RNAseq_with_doublingrate_797gene.csv"
out_dir = "../../data/ML_v2/13_LightGBM"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load and preprocess
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X = df.drop(columns=meta_cols)
y = np.log1p(df["final_DT_mean"].astype(float))  # log-transformed target

print(f"✅ Using log-transformed doubling time (ln(DT + 1))")
print(f"Data shape: {X.shape[0]} samples × {X.shape[1]} genes")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# --- Train-test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train = {X_train.shape[0]} | Test = {X_test.shape[0]}")

# ------------------------------------------------------------
# --- LightGBM Model
# ------------------------------------------------------------
print("\n🔹 Training LightGBM (strong regularization, log target)...")

lgb_model = LGBMRegressor(
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
    n_jobs=-1,
)

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
# --- Evaluate
# ------------------------------------------------------------
y_pred_train = lgb_model.predict(X_train)
y_pred_test = lgb_model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
spearman = spearmanr(y_test, y_pred_test).correlation

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lgb_model, X_scaled, y, cv=cv, scoring="r2")
cv_mean, cv_sd = cv_scores.mean(), cv_scores.std()

print(f"""
=== LightGBM (log target) ===
Train R²       : {r2_train:.3f}
Test  R²       : {r2_test:.3f}
Test  MAE      : {mae_test:.3f}
Test  Spearman : {spearman:.3f}
5-fold CV R²   : {cv_mean:.3f} ± {cv_sd:.3f}
Best iteration : {lgb_model.best_iteration_}
""")

# ------------------------------------------------------------
# --- Save results
# ------------------------------------------------------------
summary = pd.DataFrame([{
    "Model": "LightGBM",
    "Target": "log(DT)",
    "Train_R2": r2_train,
    "Test_R2": r2_test,
    "MAE": mae_test,
    "Spearman": spearman,
    "CV_R2_mean": cv_mean,
    "CV_R2_SD": cv_sd,
    "Best_iteration": lgb_model.best_iteration_
}])

summary_out = os.path.join(out_dir, "LightGBM_summary_log.csv")
summary.to_csv(summary_out, index=False)
print(f"✅ Saved summary → {summary_out}")

model_out = os.path.join(out_dir, "LightGBM_logDT_model.pkl")
joblib.dump(lgb_model, model_out)
print(f"💾 Saved trained LightGBM model → {model_out}")

# ------------------------------------------------------------
# --- Plot: observed vs predicted
# ------------------------------------------------------------
plt.figure(figsize=(4.5, 4))
sns.regplot(
    x=y_test,
    y=y_pred_test,
    scatter_kws={'s': 50, 'alpha': 0.8},
    line_kws={'color': 'steelblue'}
)
plt.xlabel("Observed log(DT+1)")
plt.ylabel("Predicted (LightGBM)")
plt.title(f"Observed vs Predicted (R²={r2_test:.2f}, MAE={mae_test:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "LightGBM_scatter_log.png"), dpi=150)
plt.close()

# ------------------------------------------------------------
# --- SHAP Interpretation
# ------------------------------------------------------------
if HAS_SHAP:
    try:
        print("\n🔍 Running SHAP interpretation...")
        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        explainer = shap.Explainer(lgb_model, X_train_df)
        shap_values = explainer(X_train_df)

        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "LightGBM_SHAP_beeswarm.png"), dpi=150)
        plt.close()

        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "LightGBM_SHAP_bar.png"), dpi=150)
        plt.close()

        print("✅ SHAP analysis complete.")
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")
else:
    print("ℹ️ SHAP not installed; skipping interpretation.")

print("\n🎯 Done. Outputs written to:", out_dir)
