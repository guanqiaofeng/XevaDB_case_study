#!/usr/bin/env python3
"""
14_train_ElasticNet.py

Final ElasticNet model using log-transformed doubling time (ln(DT + 1)) target.
NOW UPDATED TO USE TOP 500 HIGH-VARIANCE GENES.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

# Optional dependencies
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import gseapy as gp
    HAS_GSEAPY = True
except Exception:
    HAS_GSEAPY = False

# ------------------------------------------------------------
# --- Setup
# ------------------------------------------------------------
sns.set(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams.update({"figure.dpi": 150})

in_file = "../../data/analyze/RNAseq_with_doublingrate_797gene.csv"
out_dir = "../../data/ML_v2/14_ElasticNet_500HVG"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load and preprocess data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X_full = df.drop(columns=meta_cols)
y = np.log1p(df["final_DT_mean"].astype(float))

print(f"Loaded full dataset: {X_full.shape[0]} samples × {X_full.shape[1]} genes")

# ------------------------------------------------------------
# --- NEW: Select top 500 HVGs
# ------------------------------------------------------------
n_genes = 500
var = X_full.var(axis=0)
top_genes = var.sort_values(ascending=False).head(n_genes).index.tolist()

X = X_full[top_genes]

print(f"🔬 Using top {n_genes} high-variance genes")
print(f"Selected matrix: {X.shape[0]} samples × {X.shape[1]} HVGs")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# --- Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# --- ElasticNet model
# ------------------------------------------------------------
model = ElasticNetCV(
    alphas=np.logspace(-4, 2, 60),
    l1_ratio=np.linspace(0.1, 1.0, 10),
    cv=5,
    n_jobs=-1,
    max_iter=50000,
    tol=1e-5,
    random_state=42
)

print("\n🔹 Training ElasticNetCV on log-transformed target (Top 500 HVGs)...")
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ------------------------------------------------------------
# --- Metrics
# ------------------------------------------------------------
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
spearman = spearmanr(y_test, y_pred_test).correlation

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
cv_mean, cv_sd = cv_scores.mean(), cv_scores.std()

print(f"""
=== ElasticNetCV (log target, Top 500 HVGs) ===
Best alpha     : {model.alpha_:.6f}
Best l1_ratio  : {model.l1_ratio_:.3f}
Train R²       : {r2_train:.3f}
Test  R²       : {r2_test:.3f}
Test  MAE      : {mae_test:.3f}
Test  Spearman : {spearman:.3f}
5-fold CV R²   : {cv_mean:.3f} ± {cv_sd:.3f}
""")

# ------------------------------------------------------------
# --- Save results
# ------------------------------------------------------------
coef = pd.Series(model.coef_, index=X.columns)
coef_nonzero = coef[coef != 0].sort_values(key=np.abs, ascending=False)
coef_nonzero.to_csv(os.path.join(out_dir, "ElasticNet_coefficients_500HVG.csv"))

summary = pd.DataFrame([{
    "Target": "log(DT)",
    "HVGs": n_genes,
    "Train_R2": r2_train,
    "Test_R2": r2_test,
    "MAE": mae_test,
    "Spearman": spearman,
    "CV_R2_mean": cv_mean,
    "CV_R2_SD": cv_sd,
    "Best_alpha": model.alpha_,
    "Best_l1_ratio": model.l1_ratio_,
    "nGenes_nonzero": len(coef_nonzero)
}])
summary_out = os.path.join(out_dir, "ElasticNet_summary_500HVG.csv")
summary.to_csv(summary_out, index=False)
print(f"✅ Saved summary → {summary_out}")

# Save trained model
model_out = os.path.join(out_dir, "ElasticNet_model_500HVG.pkl")
joblib.dump(model, model_out)
print(f"💾 Saved trained ElasticNet model → {model_out}")

# ------------------------------------------------------------
# --- Plots
# ------------------------------------------------------------
plt.figure(figsize=(4.5, 4))
sns.regplot(
    x=y_test,
    y=y_pred_test,
    scatter_kws={'s': 50, 'alpha': 0.8},
    line_kws={'color': 'steelblue'},
    ci=95,
    color='skyblue'
)
plt.xlabel("Observed log(final_DT_mean + 1) (Test)")
plt.ylabel("Predicted (ElasticNet)")
plt.title(f"Observed vs Predicted (R²={r2_test:.2f}, MAE={mae_test:.2f})\nTop 500 HVGs")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ElasticNet_scatter_500HVG.png"), dpi=150)
plt.close()

# Top 25 genes
top25 = coef_nonzero.head(25).sort_values()
plt.figure(figsize=(6,6))
top25.plot(kind="barh", color="teal")
plt.title("Top 25 ElasticNet Coefficients (Top 500 HVGs)")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ElasticNet_top25_500HVG.png"), dpi=150)
plt.close()

# ------------------------------------------------------------
# --- SHAP analysis
# ------------------------------------------------------------
shap_top_genes = None
if HAS_SHAP:
    try:
        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        X_test_df = pd.DataFrame(X_test, columns=X.columns)

        explainer = shap.Explainer(model, X_train_df, algorithm="linear")
        shap_values = explainer.shap_values(X_test_df)

        shap_df = pd.DataFrame(shap_values, columns=X.columns)
        shap_df.to_csv(os.path.join(out_dir, "ElasticNet_SHAP_values_500HVG.csv"), index=False)

        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        shap_top_genes = mean_abs_shap.head(50).index.tolist()
        pd.Series(shap_top_genes).to_csv(os.path.join(out_dir, "ElasticNet_SHAP_top50_500HVG.csv"), index=False)

        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ElasticNet_SHAP_summary_500HVG.png"), dpi=150)
        plt.close()

        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ElasticNet_SHAP_bar_500HVG.png"), dpi=150)
        plt.close()

        print("✅ SHAP analysis complete.")
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")

print("\n🎯 Done. Outputs written to:", out_dir)
