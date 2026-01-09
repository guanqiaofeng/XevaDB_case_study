#!/usr/bin/env python3
"""
15_run_multi_HVG_experiment.py

Automatically runs ElasticNet, LightGBM, and CatBoost
across multiple HVG (top-variance gene) thresholds.

Outputs:
 - unified CSV of all results
 - per-model scatter plots
 - heatmap of performance vs #genes
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import ElasticNetCV
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# ------------------------------------------------------------
# --- Settings
# ------------------------------------------------------------
HVG_LIST = [100, 200, 300, 400, 500]
in_file = "../../data/analyze/RNAseq_with_doublingrate_797gene.csv"
out_dir = "../../data/ML_4model/15_multiHVG"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# --- Load base data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})
meta_cols = ["modelID", "final_DT_mean", "final_DT_sd", "n_reps", "mean_r2"]

X_full = df.drop(columns=meta_cols)
y_full = np.log1p(df["final_DT_mean"].astype(float))

print(f"Loaded: {X_full.shape[0]} samples × {X_full.shape[1]} genes")

# ------------------------------------------------------------
# --- Helper functions
# ------------------------------------------------------------
def evaluate_metrics(y_train, y_test, y_pred_train, y_pred_test):
    return {
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Test_Spearman": spearmanr(y_test, y_pred_test).correlation
    }


def plot_scatter(y_true, y_pred, title, fname):
    plt.figure(figsize=(4.2, 4))
    sns.regplot(
        x=y_true, y=y_pred,
        scatter_kws={'s': 45, 'alpha': 0.75},
        line_kws={'color': 'steelblue'}
    )
    plt.xlabel("Observed log(DT+1)")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ------------------------------------------------------------
# --- Lassoed Forest (Haibe-Kains & Tibshirani)
# ------------------------------------------------------------

class LassoedForest:
    def __init__(self, 
                 rf_trees=2000,
                 rf_max_depth=None,
                 rf_min_samples_split=4,
                 rf_min_samples_leaf=2,
                 gamma=1.0,
                 random_state=42):
        self.rf_trees = rf_trees
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.gamma = gamma
        self.random_state = random_state
        
        # models
        self.rf = None
        self.lasso = None
        self.feature_weights = None

    def _compute_rf_usage(self, rf, X):
        """Compute depth-weighted feature usage based on actual splits."""
        n_features = X.shape[1]
        usage = np.zeros(n_features)

        for tree in rf.estimators_:
            tree_struct = tree.tree_
            feature = tree_struct.feature
            children_left = tree_struct.children_left
            children_right = tree_struct.children_right

            def traverse(node, depth):
                if feature[node] >= 0:
                    usage[feature[node]] += 1.0 / (depth + 1)
                    traverse(children_left[node], depth + 1)
                    traverse(children_right[node], depth + 1)

            traverse(0, 0)

        return usage

    def fit(self, X, y):
        # 1. Fit Random Forest
        self.rf = RandomForestRegressor(
            n_estimators=self.rf_trees,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            min_samples_leaf=self.rf_min_samples_leaf,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
            bootstrap=True
        )
        self.rf.fit(X, y)

        # 2. Compute feature usage importance
        usage = self._compute_rf_usage(self.rf, X)

        # Small constant to avoid divide-by-zero
        eps = 1e-6
        self.feature_weights = 1.0 / ((usage ** self.gamma) + eps)

        # 3. Reweight X for adaptive Lasso
        X_weighted = X / self.feature_weights

        # 4. Fit LassoCV on reweighted features
        self.lasso = LassoCV(
            cv=5,
            n_jobs=-1,
            random_state=self.random_state,
            max_iter=50000
        )
        self.lasso.fit(X_weighted, y)

    def predict(self, X):
        # Weight input the same way
        X_weighted = X / self.feature_weights
        return self.lasso.predict(X_weighted)

    def get_coefficients(self, feature_names):
        coef = pd.Series(self.lasso.coef_, index=feature_names)
        return coef[coef != 0].sort_values(ascending=False)



# ------------------------------------------------------------
# --- Storage for final summary
# ------------------------------------------------------------
rows = []

# ------------------------------------------------------------
# --- Loop over HVG settings
# ------------------------------------------------------------
for n_genes in HVG_LIST:
    print("\n" + "="*70)
    print(f"🔬 Running models for top {n_genes} HVGs")
    print("="*70)

    # Select HVGs
    var = X_full.var(axis=0)
    top_genes = var.sort_values(ascending=False).head(n_genes).index.tolist()
    X = X_full[top_genes]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_full, test_size=0.2, random_state=42
    )

    # KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # --------------------------------------------------------
    # === ElasticNet ===
    # --------------------------------------------------------
    print("\nElasticNet...")
    enet = ElasticNetCV(
        alphas=np.logspace(-4, 2, 60),
        l1_ratio=np.linspace(0.1, 1.0, 10),
        cv=5, n_jobs=-1, max_iter=50000, random_state=42
    )
    enet.fit(X_train, y_train)

    en_pred_train = enet.predict(X_train)
    en_pred_test = enet.predict(X_test)

    en_metrics = evaluate_metrics(y_train, y_test, en_pred_train, en_pred_test)
    en_metrics["CV_R2_mean"] = cross_val_score(enet, X_scaled, y_full, cv=cv, scoring="r2").mean()
    en_metrics["CV_R2_SD"] = cross_val_score(enet, X_scaled, y_full, cv=cv, scoring="r2").std()

    rows.append({"Model": "ElasticNet", "n_genes": n_genes, **en_metrics})

    plot_scatter(
        y_test, en_pred_test,
        f"ElasticNet (top {n_genes})",
        os.path.join(out_dir, f"EN_scatter_top{n_genes}.png")
    )

    # --------------------------------------------------------
    # === LightGBM ===
    # --------------------------------------------------------
    print("\nLightGBM...")
    lgb = LGBMRegressor(
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
        n_jobs=-1
    )

    lgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="l2",
        callbacks=[early_stopping(stopping_rounds=100),
                   log_evaluation(period=50)]
    )

    lgb_pred_train = lgb.predict(X_train)
    lgb_pred_test  = lgb.predict(X_test)

    lgb_metrics = evaluate_metrics(y_train, y_test, lgb_pred_train, lgb_pred_test)
    cv_scores = cross_val_score(lgb, X_scaled, y_full, cv=cv, scoring="r2")
    lgb_metrics["CV_R2_mean"] = cv_scores.mean()
    lgb_metrics["CV_R2_SD"] = cv_scores.std()
    lgb_metrics["Best_iter"] = lgb.best_iteration_

    rows.append({"Model": "LightGBM", "n_genes": n_genes, **lgb_metrics})

    plot_scatter(
        y_test, lgb_pred_test,
        f"LightGBM (top {n_genes})",
        os.path.join(out_dir, f"LGBM_scatter_top{n_genes}.png")
    )

    # --------------------------------------------------------
    # === Random Forest (instead of CatBoost) ===
    # --------------------------------------------------------
    print("\nRandomForest...")

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=2000,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",   # strong regularization for small n
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_pred_train = rf.predict(X_train)
    rf_pred_test  = rf.predict(X_test)

    rf_metrics = evaluate_metrics(y_train, y_test, rf_pred_train, rf_pred_test)

    cv_scores = cross_val_score(rf, X_scaled, y_full, cv=cv, scoring="r2")
    rf_metrics["CV_R2_mean"] = cv_scores.mean()
    rf_metrics["CV_R2_SD"] = cv_scores.std()
    rf_metrics["Best_iter"] = np.nan  # RF does not have best iteration

    rows.append({"Model": "RandomForest", "n_genes": n_genes, **rf_metrics})

    plot_scatter(
        y_test, rf_pred_test,
        f"RandomForest (top {n_genes})",
        os.path.join(out_dir, f"RF_scatter_top{n_genes}.png")
    )

    # --------------------------------------------------------
    # === Lassoed Forest ===
    # --------------------------------------------------------
    print("\nLassoed Forest...")

    lf = LassoedForest(
        rf_trees=2000,
        rf_max_depth=None,
        rf_min_samples_split=4,
        rf_min_samples_leaf=2,
        gamma=1.0,
        random_state=42
    )
    lf.fit(X_train, y_train)

    lf_pred_train = lf.predict(X_train)
    lf_pred_test  = lf.predict(X_test)

    lf_metrics = evaluate_metrics(y_train, y_test, lf_pred_train, lf_pred_test)

    cv_scores = cross_val_score(lf.lasso, X_scaled / lf.feature_weights, y_full, cv=cv, scoring="r2")
    lf_metrics["CV_R2_mean"] = cv_scores.mean()
    lf_metrics["CV_R2_SD"] = cv_scores.std()
    lf_metrics["Best_iter"] = np.nan

    rows.append({"Model": "LassoedForest", "n_genes": n_genes, **lf_metrics})

    plot_scatter(
        y_test, lf_pred_test,
        f"LassoedForest (top {n_genes})",
        os.path.join(out_dir, f"LF_scatter_top{n_genes}.png")
    )

# ------------------------------------------------------------
# --- Save unified results
# ------------------------------------------------------------
summary_df = pd.DataFrame(rows)
summary_out = os.path.join(out_dir, "multiHVG_summary.csv")
summary_df.to_csv(summary_out, index=False)

print("\n📁 Saved unified summary →", summary_out)

# ------------------------------------------------------------
# --- Heatmap (Test R² vs #genes)
# ------------------------------------------------------------
pivot = summary_df.pivot(index="Model", columns="n_genes", values="Test_R2")

plt.figure(figsize=(8, 4))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
plt.title("Test R² across HVG settings")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "heatmap_TestR2.png"), dpi=150)
plt.close()

print("🎯 Done. Outputs saved to:", out_dir)
