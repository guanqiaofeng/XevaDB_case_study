#!/usr/bin/env python3
"""
11_train_ML_doublingrate.py

Improved version of baseline ML model:
Predicts PDX doubling time from RNA-seq data with robust feature filtering,
weighted regression, and proper cross-validation.

Key improvements:
 - Pre-filter top-variance genes (reduce p >> n instability)
 - Weighted ElasticNetCV with stronger regularization
 - Cross-validation that respects sample weights
 - Permutation sanity check
 - Ridge/Lasso comparison and overlap report
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# --- Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_filtered.csv"
out_dir = "../../data/ML"
os.makedirs(out_dir, exist_ok=True)

coef_out = os.path.join(out_dir, "ElasticNet_gene_coefficients_v11.csv")
metrics_out = os.path.join(out_dir, "ML_model_performance_summary_v11.csv")

# ------------------------------------------------------------
# --- Load data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]
X = df.drop(columns=meta_cols)
y = df["final_DT_mean"]
sd = df["final_DT_sd"].copy()

# ------------------------------------------------------------
# --- Handle missing / unreliable SD for weighting
# ------------------------------------------------------------
sd = sd.replace(0, np.nan)
weights = 1 / (sd + 1e-3)
min_weight = np.nanpercentile(weights, 10)
weights = weights.fillna(min_weight)
weights = weights / weights.mean()
print(f"Sample weights range: {weights.min():.3f} - {weights.max():.3f}")

# ------------------------------------------------------------
# --- Feature pre-filtering: top-variance genes
# ------------------------------------------------------------
n_features_keep = 3000
gene_vars = X.var().sort_values(ascending=False)
top_genes = gene_vars.head(n_features_keep).index
X = X[top_genes]
print(f"Selected top {len(top_genes)} high-variance genes for modeling.")

# ------------------------------------------------------------
# --- Standardize features
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# --- Cross-validation setup
# ------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------------------------------
# --- Train ElasticNetCV (weighted)
# ------------------------------------------------------------
print("\nTraining ElasticNetCV (robust settings)...")
elastic = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7],
    n_alphas=200,
    cv=cv,
    n_jobs=-1,
    random_state=42,
    max_iter=10000
)
elastic.fit(X_scaled, y, sample_weight=weights)

y_pred_el = elastic.predict(X_scaled)
r2_el = r2_score(y, y_pred_el)
mae_el = mean_absolute_error(y, y_pred_el)
sp_el = spearmanr(y, y_pred_el).correlation
print(f"ElasticNetCV: R²={r2_el:.3f}, MAE={mae_el:.2f}, Spearman={sp_el:.3f}")

# Save coefficients
coef = pd.Series(elastic.coef_, index=X.columns)
coef = coef[coef != 0].sort_values(ascending=False)
coef.to_csv(coef_out)
print(f"Saved {len(coef)} non-zero gene coefficients → {coef_out}")

# ------------------------------------------------------------
# --- Ridge and Lasso baselines
# ------------------------------------------------------------
print("\nTraining RidgeCV...")
ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=cv)
ridge.fit(X_scaled, y, sample_weight=weights)
y_pred_ridge = ridge.predict(X_scaled)
r2_ridge = r2_score(y, y_pred_ridge)
mae_ridge = mean_absolute_error(y, y_pred_ridge)
sp_ridge = spearmanr(y, y_pred_ridge).correlation
print(f"RidgeCV: R²={r2_ridge:.3f}, MAE={mae_ridge:.2f}, Spearman={sp_ridge:.3f}")

print("\nTraining LassoCV...")
lasso = LassoCV(cv=cv, n_jobs=-1, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y, sample_weight=weights)
y_pred_lasso = lasso.predict(X_scaled)
r2_lasso = r2_score(y, y_pred_lasso)
mae_lasso = mean_absolute_error(y, y_pred_lasso)
sp_lasso = spearmanr(y, y_pred_lasso).correlation
print(f"LassoCV: R²={r2_lasso:.3f}, MAE={mae_lasso:.2f}, Spearman={sp_lasso:.3f}")

# ------------------------------------------------------------
# --- Summary
# ------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["ElasticNetCV", "RidgeCV", "LassoCV"],
    "R2": [r2_el, r2_ridge, r2_lasso],
    "MAE": [mae_el, mae_ridge, mae_lasso],
    "Spearman": [sp_el, sp_ridge, sp_lasso],
    "n_features_nonzero": [len(coef), X.shape[1], np.sum(lasso.coef_ != 0)]
})
results.to_csv(metrics_out, index=False)
print(f"\n✅ Saved model performance summary → {metrics_out}")

# ------------------------------------------------------------
# --- Ridge vs Lasso overlap
# ------------------------------------------------------------
ridge_top_idx = np.argsort(np.abs(ridge.coef_))[-50:]
ridge_top_genes = np.array(X.columns)[ridge_top_idx]
lasso_top_genes = np.array(X.columns)[lasso.coef_ != 0]
overlap = len(set(ridge_top_genes) & set(lasso_top_genes))
print(f"\n🔍 Overlap between Ridge (top 50) and Lasso non-zero genes: {overlap}")
if overlap > 0:
    overlap_genes = list(set(ridge_top_genes) & set(lasso_top_genes))
    pd.Series(overlap_genes, name="Gene").to_csv(
        os.path.join(out_dir, "Ridge_Lasso_topgene_overlap_v11.csv"), index=False
    )

# ------------------------------------------------------------
# --- Robustness validation
# ------------------------------------------------------------
print("\n✅ Model training complete. Proceeding to robustness validation...")

# 1️⃣ Weighted 5-fold CV (manual)
r2_scores = []
for train_idx, test_idx in cv.split(X_scaled):
    elastic.fit(X_scaled[train_idx], y.iloc[train_idx],
                sample_weight=weights.iloc[train_idx])
    y_pred = elastic.predict(X_scaled[test_idx])
    r2_scores.append(r2_score(y.iloc[test_idx], y_pred))

print(f"5-fold CV R² mean = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# 2️⃣ Permutation sanity test
np.random.seed(42)
y_perm = np.random.permutation(y)
elastic_perm = ElasticNetCV(
    l1_ratio=[0.3, 0.5],
    n_alphas=100,
    cv=cv,
    n_jobs=-1,
    random_state=42,
    max_iter=5000
)
elastic_perm.fit(X_scaled, y_perm, sample_weight=weights)
r2_perm = r2_score(y_perm, elastic_perm.predict(X_scaled))
print(f"Permutation control R² = {r2_perm:.3f} (should be near 0 or negative)")

# 3️⃣ Observed vs predicted plot
fig, ax = plt.subplots(figsize=(4.5,4))
sns.regplot(x=y, y=y_pred_el, scatter_kws={'s':40, 'alpha':0.7})
ax.set_xlabel("Observed Doubling Time (days)")
ax.set_ylabel("Predicted (ElasticNet)")
ax.set_title(f"Observed vs Predicted (R²={r2_el:.2f}, MAE={mae_el:.1f})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ElasticNet_observed_vs_predicted_v11.png"), dpi=150)
plt.close()

print("\n🎯 Training + validation complete. Ready for SHAP interpretation.")
