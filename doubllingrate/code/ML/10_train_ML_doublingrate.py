#!/usr/bin/env python3
"""
50_train_ML_doublingrate.py

Train baseline ML regressors (ElasticNet, Ridge, Lasso)
to predict PDX doubling time from RNA-seq data.

Implements weighted regression:
 - Low weight for models with high or missing SD (single-replicate)
 - Higher weight for consistent (low-SD) models

Outputs:
 - Cross-validation metrics
 - Model coefficients
 - Coefficient summary CSV
"""

import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

# ------------------------------------------------------------
# --- Paths
# ------------------------------------------------------------
in_file = "../../data/analyze/RNAseq_with_doublingrate_filtered.csv"
out_dir = "../../data/ML"
os.makedirs(out_dir, exist_ok=True)

coef_out = os.path.join(out_dir, "ElasticNet_gene_coefficients.csv")
metrics_out = os.path.join(out_dir, "ML_model_performance_summary.csv")

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
weights = weights / weights.mean()  # normalize mean weight = 1

print(f"Sample weights range: {weights.min():.3f} - {weights.max():.3f}")

# ------------------------------------------------------------
# --- Standardize features
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# --- Cross-validation setup
# ------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, weights):
    model.fit(X, y, sample_weight=weights)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    spearman = spearmanr(y, y_pred).correlation
    return r2, mae, spearman, y_pred

# ------------------------------------------------------------
# --- Train ElasticNetCV
# ------------------------------------------------------------
print("\nTraining ElasticNetCV...")
elastic = ElasticNetCV(
    l1_ratio=[.1, .3, .5, .7, .9, 1.0],
    cv=cv,
    n_jobs=-1,
    random_state=42,
    max_iter=5000
)
r2_el, mae_el, sp_el, y_pred_el = evaluate_model(elastic, X_scaled, y, weights)
print(f"ElasticNetCV: R²={r2_el:.3f}, MAE={mae_el:.2f}, Spearman={sp_el:.3f}")

# Save coefficients
coef = pd.Series(elastic.coef_, index=X.columns)
coef = coef[coef != 0].sort_values(ascending=False)
coef.to_csv(coef_out)
print(f"Saved {len(coef)} non-zero gene coefficients → {coef_out}")

# ------------------------------------------------------------
# --- Train RidgeCV (L2 only)
# ------------------------------------------------------------
print("\nTraining RidgeCV...")
ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=cv)
ridge.fit(X_scaled, y, sample_weight=weights)
y_pred_ridge = ridge.predict(X_scaled)
r2_ridge = r2_score(y, y_pred_ridge)
mae_ridge = mean_absolute_error(y, y_pred_ridge)
sp_ridge = spearmanr(y, y_pred_ridge).correlation
print(f"RidgeCV: R²={r2_ridge:.3f}, MAE={mae_ridge:.2f}, Spearman={sp_ridge:.3f}")

# ------------------------------------------------------------
# --- Train LassoCV (L1 only)
# ------------------------------------------------------------
print("\nTraining LassoCV...")
lasso = LassoCV(cv=cv, n_jobs=-1, random_state=42, max_iter=5000)
lasso.fit(X_scaled, y, sample_weight=weights)
y_pred_lasso = lasso.predict(X_scaled)
r2_lasso = r2_score(y, y_pred_lasso)
mae_lasso = mean_absolute_error(y, y_pred_lasso)
sp_lasso = spearmanr(y, y_pred_lasso).correlation
print(f"LassoCV: R²={r2_lasso:.3f}, MAE={mae_lasso:.2f}, Spearman={sp_lasso:.3f}")

# ------------------------------------------------------------
# --- Summarize results
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
# --- Visualization: observed vs predicted (ElasticNet)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5,4))
sns.regplot(x=y, y=y_pred_el, scatter_kws={'s':40, 'alpha':0.7})
ax.set_xlabel("Observed Doubling Time (days)")
ax.set_ylabel("Predicted (ElasticNet)")
ax.set_title(f"Observed vs Predicted (R²={r2_el:.2f}, MAE={mae_el:.1f})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ElasticNet_observed_vs_predicted.png"), dpi=150)
plt.close()

print("📊 Generated observed vs predicted plot.")
print("🎯 Baseline ML training complete.")

# ... overlap gene block ...

print("\n✅ Model training complete. Proceeding to robustness validation...")

# ------------------------------------------------------------
# --- Compare top gene overlap between Ridge and Lasso
# ------------------------------------------------------------

# Get top 50 absolute-weight genes from Ridge
ridge_top_idx = np.argsort(np.abs(ridge.coef_))[-50:]
ridge_top_genes = np.array(X.columns)[ridge_top_idx]

# Get all non-zero genes from Lasso
lasso_top_genes = np.array(X.columns)[lasso.coef_ != 0]

# Compute overlap
overlap = len(set(ridge_top_genes) & set(lasso_top_genes))
print(f"\n🔍 Overlap between Ridge (top 50) and Lasso non-zero genes: {overlap}")
print(f"Ridge top 50 genes: {len(ridge_top_genes)}, Lasso non-zero genes: {len(lasso_top_genes)}")

# Optionally save overlap list
overlap_genes = list(set(ridge_top_genes) & set(lasso_top_genes))
if overlap_genes:
    overlap_out = os.path.join(out_dir, "Ridge_Lasso_topgene_overlap.csv")
    pd.Series(overlap_genes, name="Gene").to_csv(overlap_out, index=False)
    print(f"✅ Saved overlapping gene list → {overlap_out}")
else:
    print("⚠️ No overlap between Ridge and Lasso top genes.")


plt.figure(figsize=(4,4))
venn2(subsets = (50, 54, 29), set_labels=('Ridge top 50', 'Lasso non-zero'))
plt.title("Overlap between Ridge and Lasso predictive genes")
plt.tight_layout()
plt.savefig("../../data/analyze/Ridge_Lasso_overlap_venn.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# --- Cross-validation robustness test
# ------------------------------------------------------------

print("\n🔍 Running 5-fold cross-validation (R² scores)...")
r2_scores = []

for train_idx, test_idx in cv.split(X_scaled):
    elastic.fit(X_scaled[train_idx], y.iloc[train_idx], sample_weight=weights.iloc[train_idx])
    y_pred = elastic.predict(X_scaled[test_idx])
    r2 = r2_score(y.iloc[test_idx], y_pred)
    r2_scores.append(r2)

print(f"Cross-val R² mean = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

print("Note: Cross-validation performed without sample weights")

# ------------------------------------------------------------
# --- Permutation test (sanity check)
# ------------------------------------------------------------
np.random.seed(42)
y_perm = np.random.permutation(y)
elastic_perm = ElasticNetCV(
    l1_ratio=[.1, .3, .5, .7, .9, 1.0],
    cv=cv,
    n_jobs=-1,
    random_state=42,
    max_iter=5000
)
elastic_perm.fit(X_scaled, y_perm, sample_weight=weights)
r2_perm = r2_score(y_perm, elastic_perm.predict(X_scaled))
print(f"Permutation control R² = {r2_perm:.3f} (should be near 0 or negative)")

# ------------------------------------------------------------
# --- Leave-one-out cross-validation (optional)
# ------------------------------------------------------------
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
r2_scores = []
for train_idx, test_idx in loo.split(X_scaled):
    elastic.fit(X_scaled[train_idx], y.iloc[train_idx], sample_weight=weights.iloc[train_idx])
    y_pred = elastic.predict(X_scaled[test_idx])
    r2_scores.append((y.iloc[test_idx].values[0], y_pred[0]))

# Compute correlation between observed and predicted across LOO
y_true, y_pred = zip(*r2_scores)
spearman_loo = spearmanr(y_true, y_pred).correlation
print(f"LOOCV Spearman correlation = {spearman_loo:.3f}")


# [Insert CV + permutation + LOO code here]
