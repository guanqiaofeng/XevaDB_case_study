#!/usr/bin/env python3
"""
13_train_CatBoost.py

CatBoost regression model for predicting log-transformed doubling time
(ln(DT + 1)) from RNA-seq gene expression (797 genes).

Mirrors ElasticNet and LightGBM pipelines:
- log(DT + 1) target
- standardized features
- strong regularization for small-n
- early stopping
- CV performance
- SHAP interpretation
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

from catboost import CatBoostRegressor, Pool

# ------------------------------------------------------------
# --- Setup
# ------------------------------------------------------------
sns.set(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams.update({"figure.dpi": 150})

in_file = "../../data/analyze/RNAseq_with_doublingrate_797gene.csv"
out_dir = "../../data/ML_CatBoost"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load & preprocess
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X = df.drop(columns=meta_cols)
y = np.log1p(df["final_DT_mean"].astype(float))

print(f"Using log-transformed target ln(DT+1)")
print(f"Data shape: {X.shape[0]} samples × {X.shape[1]} genes")

# Standardization is recommended even though CatBoost can handle raw features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# --- Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train = {X_train.shape[0]} | Test = {X_test.shape[0]}")

train_pool = Pool(X_train, y_train)
test_pool  = Pool(X_test,  y_test)

# ------------------------------------------------------------
# --- CatBoost model
# ------------------------------------------------------------
print("\n🔹 Training CatBoost (strong regularization)...")

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    loss_function="RMSE",
    random_seed=42,
    task_type="CPU",
    od_type="Iter",
    od_wait=100,
    verbose=False
)

model.fit(train_pool, eval_set=test_pool)

# ------------------------------------------------------------
# --- Evaluate
# ------------------------------------------------------------
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
spearman_corr = spearmanr(y_test, y_pred_test).correlation

# 5-fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
cv_mean, cv_sd = cv_scores.mean(), cv_scores.std()

print(f"""
=== CatBoost (log target) ===
Train R²       : {r2_train:.3f}
Test  R²       : {r2_test:.3f}
Test  MAE      : {mae_test:.3f}
Test  Spearman : {spearman_corr:.3f}
5-fold CV R²   : {cv_mean:.3f} ± {cv_sd:.3f}
Best iteration : {model.get_best_iteration()}
""")

# ------------------------------------------------------------
# --- Save summary
# ------------------------------------------------------------
summary = pd.DataFrame([{
    "Model": "CatBoost",
    "Target": "log(DT)",
    "Train_R2": r2_train,
    "Test_R2": r2_test,
    "MAE": mae_test,
    "Spearman": spearman_corr,
    "CV_R2_mean": cv_mean,
    "CV_R2_SD": cv_sd,
    "Best_iteration": model.get_best_iteration()
}])

summary_out = os.path.join(out_dir, "CatBoost_summary_log.csv")
summary.to_csv(summary_out, index=False)

model_out = os.path.join(out_dir, "CatBoost_logDT_model.cbm")
model.save_model(model_out)

print(f"Saved summary → {summary_out}")
print(f"Saved CatBoost model → {model_out}")

# ------------------------------------------------------------
# --- Plot: Observed vs predicted
# ------------------------------------------------------------
plt.figure(figsize=(4.5,4))
sns.regplot(
    x=y_test,
    y=y_pred_test,
    scatter_kws={'s':50,'alpha':0.8},
    line_kws={'color':'steelblue'}
)
plt.xlabel("Observed log(DT+1)")
plt.ylabel("Predicted (CatBoost)")
plt.title(f"Observed vs Predicted (R²={r2_test:.2f}, MAE={mae_test:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "CatBoost_scatter_log.png"), dpi=150)
plt.close()

# ------------------------------------------------------------
# --- SHAP (built-in CatBoost SHAP)
# ------------------------------------------------------------
try:
    print("\n🔍 Running CatBoost SHAP...")

    shap_vals = model.get_feature_importance(train_pool, type="ShapValues")
    shap_values = shap_vals[:, :-1]   # last column = expected value

    # mean abs SHAP
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_importance = shap_df.abs().mean().sort_values(ascending=False)
    shap_importance.to_csv(os.path.join(out_dir, "CatBoost_SHAP_importance.csv"))

    # Top 25 barplot
    top25 = shap_importance.head(25).sort_values()
    plt.figure(figsize=(6,6))
    top25.plot(kind="barh", color="purple")
    plt.title("Top 25 SHAP Features (CatBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "CatBoost_SHAP_top25.png"), dpi=150)
    plt.close()

    print("✅ SHAP analysis complete.")

except Exception as e:
    print(f"⚠️ SHAP failed: {e}")

print("\n🎯 Done. Outputs written to:", out_dir)
