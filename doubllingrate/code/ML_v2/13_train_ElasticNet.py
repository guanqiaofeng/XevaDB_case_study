#!/usr/bin/env python3
"""
13_train_ElasticNet.py

Final ElasticNet model using log-transformed doubling time (ln(DT + 1)) target.

Builds upon 12_train_ElasticNet_tuned.py with:
- SHAP interpretation (global + bar plots)
- Enrichment analysis (GO Biological Process 2023 + KEGG 2021 Human)
- Visual outputs for PPT/manuscript
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
import seaborn as sns

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
out_dir = "../../data/ML_v2/13"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load and preprocess data
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

print("\n🔹 Training ElasticNetCV on log-transformed target ...")
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
=== ElasticNetCV (log target) ===
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
coef_nonzero.to_csv(os.path.join(out_dir, "ElasticNet_coefficients_log.csv"))

summary = pd.DataFrame([{
    "Target": "log(DT)",
    "Train_R2": r2_train,
    "Test_R2": r2_test,
    "MAE": mae_test,
    "Spearman": spearman,
    "CV_R2_mean": cv_mean,
    "CV_R2_SD": cv_sd,
    "Best_alpha": model.alpha_,
    "Best_l1_ratio": model.l1_ratio_,
    "nGenes": len(coef_nonzero)
}])
summary_out = os.path.join(out_dir, "ElasticNet_summary_log.csv")
summary.to_csv(summary_out, index=False)
print(f"✅ Saved summary → {summary_out}")

# Save trained ElasticNet model
model_out = os.path.join(out_dir, "ElasticNet_logDT_model.pkl")
joblib.dump(model, model_out)
print(f"✅ Saved trained ElasticNet model → {model_out}")

# ------------------------------------------------------------
# --- Plots
# ------------------------------------------------------------
# Scatter
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
plt.title(f"Test Set: Observed vs Predicted (R²={r2_test:.2f}, MAE={mae_test:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ElasticNet_scatter_log.png"), dpi=150)
plt.close()

# Top 25 genes
top25 = coef_nonzero.head(25).sort_values()
plt.figure(figsize=(6,6))
top25.plot(kind="barh", color="teal")
plt.title("Top 25 ElasticNet Coefficients (log target)")
plt.xlabel("Coefficient (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ElasticNet_top25_log.png"), dpi=150)
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
        shap_df.to_csv(os.path.join(out_dir, "ElasticNet_SHAP_values.csv"), index=False)

        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        shap_top_genes = mean_abs_shap.head(50).index.tolist()
        pd.Series(shap_top_genes, name="Gene").to_csv(os.path.join(out_dir, "ElasticNet_SHAP_top50.csv"), index=False)

        shap.summary_plot(shap_values, X_test_df, feature_names=X.columns, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ElasticNet_SHAP_summary.png"), dpi=150)
        plt.close()

        shap.summary_plot(shap_values, X_test_df, feature_names=X.columns, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ElasticNet_SHAP_bar.png"), dpi=150)
        plt.close()

        print(f"✅ SHAP analysis done. Top-50 genes saved.")
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
else:
    print("ℹ️ SHAP not installed; skipping interpretation.")

# ------------------------------------------------------------
# --- Enrichment via gseapy
# ------------------------------------------------------------
if HAS_GSEAPY and shap_top_genes is not None:
    try:
        enrich_frames = []
        for lib in ["GO_Biological_Process_2023", "KEGG_2021_Human"]:
            enr = gp.enrichr(
                gene_list=shap_top_genes,
                gene_sets=[lib],
                organism="Human",
                outdir=None,
                cutoff=1.0
            )
            res = enr.results.copy()
            res["library"] = lib
            enrich_frames.append(res[["Term", "Adjusted P-value", "Combined Score", "library"]])
        
        enrich_all = pd.concat(enrich_frames)
        enrich_out = os.path.join(out_dir, "ElasticNet_SHAP_enrichment.csv")
        enrich_all.to_csv(enrich_out, index=False)
        print(f"✅ Saved enrichment summary → {enrich_out}")

        # Visualization: top 10 enriched terms
        plt.figure(figsize=(7,5))
        sns.barplot(
            data=enrich_all.groupby("library").head(10),
            x="Combined Score", y="Term", hue="library"
        )
        plt.title("Top 10 Enriched GO/KEGG Terms (SHAP Top 50)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ElasticNet_SHAP_enrichment_bar.png"), dpi=150)
        plt.close()

    except Exception as e:
        print(f"⚠️ Enrichment analysis failed: {e}")
else:
    print("ℹ️ gseapy not installed or SHAP top genes unavailable; skipping enrichment.")

print("\n🎯 Done. Outputs written to:", out_dir)
