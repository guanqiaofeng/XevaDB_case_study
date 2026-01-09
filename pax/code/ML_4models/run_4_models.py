#!/usr/bin/env python3
"""
pax_4model_HVG600_reproducible.py

Reproduce 4-model classification baseline for paclitaxel response:
  - ElasticNet Logistic Regression
  - LightGBM
  - RandomForest
  - Lassoed Forest (RF leaf encoding + L1 logistic)

Uses:
  - Input: RNAseq_with_pax_800gene.csv
  - Target: pax_cat (1=CR/PR responder, 0=SD/PD)
  - HVG selection: top 600 most variable genes (training-only to avoid leakage)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
in_file = "../../data/procdata/RNAseq_with_pax_800gene.csv"
out_dir = "../../data/ML_4model/PAX_HVG600_reproduce"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(in_file, dtype={"modelID": str})

meta_cols = ["modelID", "pax_mRECIST", "pax_cat", "pax_ord"]
X_full = df.drop(columns=meta_cols)
y_full = df["pax_cat"].astype(int).values

print(f"Loaded matrix: {X_full.shape[0]} samples × {X_full.shape[1]} genes")
print("Class balance:", dict(pd.Series(y_full).value_counts().sort_index()))

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def select_hvg(X_train_df, n=600):
    # HVG selection must be fit ONLY on training data to avoid leakage
    var = X_train_df.var(axis=0)
    return var.sort_values(ascending=False).head(n).index.tolist()

def metrics_binary(y_true, y_prob, y_pred):
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "ACC": accuracy_score(y_true, y_pred),
        "BACC": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }

def save_confmat(cm, labels, title, path):
    fig = plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# ------------------------------------------------------------
# CV setup (recommended for N=68)
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

# ------------------------------------------------------------
# 1) ElasticNet Logistic Regression
# ------------------------------------------------------------
print("\n=== 1) ElasticNet Logistic (HVG600, CV) ===")
fold_metrics = []
for fold, (tr, te) in enumerate(skf.split(X_full, y_full), 1):
    Xtr_df, Xte_df = X_full.iloc[tr], X_full.iloc[te]
    ytr, yte = y_full[tr], y_full[te]

    top600 = select_hvg(Xtr_df, n=600)
    Xtr, Xte = Xtr_df[top600].values, Xte_df[top600].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=20000,
            class_weight=None,
            random_state=42
        ))
    ])

    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    fold_metrics.append(metrics_binary(yte, prob, pred))

m = pd.DataFrame(fold_metrics).mean().to_dict()
results.append({"Model": "ElasticNetLogistic", **m})
print(m)

# ------------------------------------------------------------
# 2) LightGBM
# ------------------------------------------------------------
print("\n=== 2) LightGBM (HVG600, CV) ===")
fold_metrics = []
for fold, (tr, te) in enumerate(skf.split(X_full, y_full), 1):
    Xtr_df, Xte_df = X_full.iloc[tr], X_full.iloc[te]
    ytr, yte = y_full[tr], y_full[te]

    top600 = select_hvg(Xtr_df, n=600)
    Xtr, Xte = Xtr_df[top600].values, Xte_df[top600].values

    clf = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=3,
        num_leaves=8,
        min_child_samples=10,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        n_jobs=-1
    )

    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    fold_metrics.append(metrics_binary(yte, prob, pred))

m = pd.DataFrame(fold_metrics).mean().to_dict()
results.append({"Model": "LightGBM", **m})
print(m)

# ------------------------------------------------------------
# 3) RandomForest
# ------------------------------------------------------------
print("\n=== 3) RandomForest (HVG600, CV) ===")
fold_metrics = []
for fold, (tr, te) in enumerate(skf.split(X_full, y_full), 1):
    Xtr_df, Xte_df = X_full.iloc[tr], X_full.iloc[te]
    ytr, yte = y_full[tr], y_full[te]

    top600 = select_hvg(Xtr_df, n=600)
    Xtr, Xte = Xtr_df[top600].values, Xte_df[top600].values

    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=4,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    fold_metrics.append(metrics_binary(yte, prob, pred))

m = pd.DataFrame(fold_metrics).mean().to_dict()
results.append({"Model": "RandomForest", **m})
print(m)

# ------------------------------------------------------------
# 4) Lassoed Forest (RF leaf encoding + L1 Logistic)
# ------------------------------------------------------------
print("\n=== 4) Lassoed Forest (HVG600, CV) ===")
fold_metrics = []
for fold, (tr, te) in enumerate(skf.split(X_full, y_full), 1):
    Xtr_df, Xte_df = X_full.iloc[tr], X_full.iloc[te]
    ytr, yte = y_full[tr], y_full[te]

    top600 = select_hvg(Xtr_df, n=600)
    Xtr, Xte = Xtr_df[top600].values, Xte_df[top600].values

    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=4,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(Xtr, ytr)

    leaves_tr = rf.apply(Xtr)  # (n_samples, n_trees)
    leaves_te = rf.apply(Xte)

    # L1 logistic on leaf IDs (treat as categorical via one-hot-ish trick)
    # Simple approach: use integer leaf ids directly (works but not ideal).
    # Better: OneHotEncode leaves (recommended); keeping simple here to mirror your workflow.
    lasso_logit = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0,
        max_iter=20000,
        random_state=42
    )
    lasso_logit.fit(leaves_tr, ytr)

    prob = lasso_logit.predict_proba(leaves_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    fold_metrics.append(metrics_binary(yte, prob, pred))

m = pd.DataFrame(fold_metrics).mean().to_dict()
results.append({"Model": "LassoedForest", **m})
print(m)

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------
summary_df = pd.DataFrame(results)
summary_path = os.path.join(out_dir, "PAX_HVG600_model_summary.csv")
summary_df.to_csv(summary_path, index=False, float_format="%.4f")
print("\n📁 Saved summary →", summary_path)
print("All outputs saved in:", out_dir)
