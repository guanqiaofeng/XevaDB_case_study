#!/usr/bin/env python3
"""
12_train_ElasticNet_tuned.py

Compare ElasticNetCV model performance on 3 target transformations:
  - Raw Doubling Time
  - Log-transformed (ln(DT))
  - Z-scored DT

Uses same train/test split and reports R², MAE, Spearman, CV results.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

# ------------------------------------------------------------
# --- Setup
# ------------------------------------------------------------
sns.set(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams.update({"figure.dpi": 150})

in_file = "../../data/analyze/RNAseq_with_doublingrate_797gene.csv"
out_dir = "../../data/ML_v2/12"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load and prepare data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]
X = df.drop(columns=meta_cols)
y_raw = df["final_DT_mean"].astype(float)

# Target variants
y_log = np.log1p(y_raw)
y_z = (y_raw - y_raw.mean()) / y_raw.std()
target_variants = {
    "Raw": y_raw,
    "Log": y_log,
    "Zscore": y_z
}

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# --- Train/test split (same indices for all targets)
# ------------------------------------------------------------
X_train, X_test, idx_train, idx_test = train_test_split(
    X_scaled, np.arange(len(y_raw)), test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# --- Function to train & evaluate ElasticNet
# ------------------------------------------------------------
def train_elasticnet(X_train, X_test, y, idx_train, idx_test):
    model = ElasticNetCV(
        alphas=np.logspace(-4, 2, 60),
        l1_ratio=np.linspace(0.1, 1.0, 10),
        cv=5,
        n_jobs=-1,
        max_iter=50000,
        tol=1e-5,
        random_state=42
    )
    model.fit(X_train, y[idx_train])
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    r2_train = r2_score(y[idx_train], y_pred_train)
    r2_test = r2_score(y[idx_test], y_pred_test)
    mae_test = mean_absolute_error(y[idx_test], y_pred_test)
    spearman = spearmanr(y[idx_test], y_pred_test).correlation

    # Cross-val R²
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
    cv_mean, cv_sd = cv_scores.mean(), cv_scores.std()

    coef = pd.Series(model.coef_, index=X.columns)
    nonzero = np.sum(coef != 0)

    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_test": mae_test,
        "spearman": spearman,
        "cv_mean": cv_mean,
        "cv_sd": cv_sd,
        "n_genes": nonzero,
        "coef": coef[coef != 0],
        "y_pred_test": y_pred_test
    }

# ------------------------------------------------------------
# --- Train for each target
# ------------------------------------------------------------
results = {}
for name, y in target_variants.items():
    print(f"\n🔹 Training ElasticNetCV on {name} target ...")
    res = train_elasticnet(X_train, X_test, y, idx_train, idx_test)
    results[name] = res

    # Save coefficients
    coef_out = os.path.join(out_dir, f"ElasticNet_coefficients_{name.lower()}.csv")
    res["coef"].sort_values(ascending=False).to_csv(coef_out)
    
    # Scatter plot (standardized y-axis)
    plt.figure(figsize=(4,4))
    plt.scatter(y[idx_test], res["y_pred_test"], alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel(f"Observed {name} DT")
    plt.ylabel(f"Predicted {name} DT")
    plt.title(f"{name}: Test R²={res['r2_test']:.2f}, Spearman={res['spearman']:.2f}")
    plt.xlim(y.min(), y.max())
    plt.ylim(y.min(), y.max())
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ElasticNet_scatter_{name.lower()}.png"), dpi=150)
    plt.close()

# ------------------------------------------------------------
# --- Summary table
# ------------------------------------------------------------
summary = pd.DataFrame([
    {
        "Target": name,
        "TrainR2": res["r2_train"],
        "TestR2": res["r2_test"],
        "MAE": res["mae_test"],
        "Spearman": res["spearman"],
        "CV_R2_mean": res["cv_mean"],
        "CV_R2_SD": res["cv_sd"],
        "nGenes": res["n_genes"]
    }
    for name, res in results.items()
])

summary_out = os.path.join(out_dir, "ElasticNet_target_comparison.csv")
summary.to_csv(summary_out, index=False, float_format="%.3f")

print("\n📊 ElasticNet Target Comparison:")
print(summary.to_string(index=False))
print(f"\n✅ Saved summary → {summary_out}")

# ------------------------------------------------------------
# --- Comparison barplot
# ------------------------------------------------------------
plt.figure(figsize=(4.5,3))
sns.barplot(data=summary, x="Target", y="TestR2", palette="coolwarm")
plt.title("ElasticNet Test R² Comparison by Target Transformation")
plt.ylabel("Test R²")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ElasticNet_R2_comparison_barplot.png"), dpi=150)
plt.close()
print("✅ Saved comparison barplot.")
