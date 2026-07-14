import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold

from pipeline_utils import (
    make_case1_paths,
    select_hvg_train_only,
    safe_spearman,
    safe_pearson,
    safe_inner_cv,
    per_tree_probability_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")

# =========================
# Configuration
# =========================
N_SPLITS = 3
N_REPEATS = 10
RANDOM_STATE = 42

N_HVG = 600
FAST_Q = 0.33
SLOW_Q = 0.67
POSITIVE_CLASS_LABEL = "slow"

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

EN_L1_RATIOS = [0.1, 0.5, 0.9]
EN_CS = [0.01, 0.1, 1.0, 10.0]
LF_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100]
LR_N_JOBS = 1
ADAPTIVE_LASSO_GAMMA = 1.0
ADAPTIVE_LASSO_EPS = 1e-3
SKIPPED_FOLD_COLUMNS = [
    "Fold",
    "Reason",
    "n_train_extreme",
    "n_test_extreme",
    "n_train_slow",
    "n_test_slow",
]


# =========================
# Helpers
# =========================


def fit_predict_elasticnet(X_train, y_train, X_test, inner_cv):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=EN_L1_RATIOS,
        Cs=EN_CS,
        cv=inner_cv,
        max_iter=20000,
        n_jobs=LR_N_JOBS,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    model.fit(X_train_scaled, y_train)

    return model.predict_proba(X_test_scaled)[:, 1], model


def fit_predict_random_forest(X_train, y_train, X_test):
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    return model.predict_proba(X_test)[:, 1], model


def fit_predict_lassoed_forest(rf_model, X_train, y_train, X_test, inner_cv):
    """
    Paper-aligned Lassoed Forest classifier.

    The random forest is first converted into a tree-prediction design matrix:
    each column is one tree's positive-class probability. A ridge logistic model
    gives initial tree weights, then a weighted L1 logistic model performs
    adaptive-lasso post-selection on the tree predictions.
    """
    Z_train = per_tree_probability_matrix(rf_model, X_train)
    Z_test = per_tree_probability_matrix(rf_model, X_test)

    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    Z_test_scaled = scaler.transform(Z_test)

    ridge = LogisticRegressionCV(
        penalty="l2",
        solver="lbfgs",
        Cs=LF_CS,
        cv=inner_cv,
        max_iter=20000,
        n_jobs=LR_N_JOBS,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    ridge.fit(Z_train_scaled, y_train)

    initial_beta = ridge.coef_.ravel()
    penalty_weights = 1.0 / (np.abs(initial_beta) + ADAPTIVE_LASSO_EPS) ** ADAPTIVE_LASSO_GAMMA

    Z_train_weighted = Z_train_scaled / penalty_weights
    Z_test_weighted = Z_test_scaled / penalty_weights

    lasso = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        Cs=LF_CS,
        cv=inner_cv,
        max_iter=20000,
        n_jobs=LR_N_JOBS,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    lasso.fit(Z_train_weighted, y_train)

    probabilities = lasso.predict_proba(Z_test_weighted)[:, 1]
    selected_trees = int(np.sum(np.abs(lasso.coef_.ravel()) > 0))

    return probabilities, ridge, lasso, selected_trees


def summarize_metric(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=0)


def compute_rf_shap(rf_model, X_train, X_test):
    """Explain held-out slow-class probabilities with interventional TreeSHAP."""
    explainer = shap.TreeExplainer(
        rf_model,
        data=X_train,
        feature_perturbation="interventional",
        model_output="probability",
    )
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    if isinstance(shap_values, list):
        shap_slow = np.asarray(shap_values[1], dtype=float)
    else:
        shap_values = np.asarray(shap_values, dtype=float)
        if shap_values.ndim == 3:
            shap_slow = shap_values[:, :, 1]
        elif shap_values.ndim == 2:
            shap_slow = shap_values
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_values.shape}")

    expected_values = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    expected_slow = float(expected_values[1] if expected_values.size > 1 else expected_values[0])
    return shap_slow, expected_slow


def make_rf_shap_records(
    fold_idx,
    repeat_idx,
    model_ids,
    genes,
    X_test,
    y_test,
    y_test_raw,
    probabilities,
    shap_slow,
    expected_slow,
):
    """Return a long table with one held-out sample-gene attribution per row."""
    n_samples, n_genes = X_test.shape
    if shap_slow.shape != (n_samples, n_genes):
        raise ValueError(
            f"SHAP shape {shap_slow.shape} does not match test matrix {(n_samples, n_genes)}"
        )

    return pd.DataFrame(
        {
            "Fold": fold_idx,
            "Repeat": repeat_idx,
            "ModelID": np.repeat(model_ids, n_genes),
            "Gene": np.tile(np.asarray(genes), n_samples),
            "SHAP_Slow": shap_slow.reshape(-1),
            "log2_expression": np.asarray(X_test, dtype=float).reshape(-1),
            "True_Label": np.repeat(np.asarray(y_test, dtype=int), n_genes),
            "Prob_Slow": np.repeat(np.asarray(probabilities, dtype=float), n_genes),
            "Expected_Prob_Slow": expected_slow,
            "log1p_doubling_time_days": np.repeat(
                np.asarray(y_test_raw, dtype=float), n_genes
            ),
        }
    )


def build_rf_shap_summary(shap_df, feature_df):
    """Summarize SHAP magnitude, direction, and fold-level feature stability."""
    shap_work = shap_df.assign(abs_SHAP=shap_df["SHAP_Slow"].abs())
    shap_summary = (
        shap_work.groupby("Gene", as_index=False)
        .agg(
            n_test_attributions=("SHAP_Slow", "size"),
            mean_abs_shap_when_selected=("abs_SHAP", "mean"),
            median_abs_shap_when_selected=("abs_SHAP", "median"),
            mean_shap_when_selected=("SHAP_Slow", "mean"),
            mean_log2_expression_when_selected=("log2_expression", "mean"),
            sum_abs_shap=("abs_SHAP", "sum"),
            sum_shap=("SHAP_Slow", "sum"),
        )
    )

    feature_summary = (
        feature_df.groupby("Gene", as_index=False)
        .agg(
            n_selected_folds=("Fold", "nunique"),
            mean_gini_importance_when_selected=("Gini_Importance", "mean"),
            median_gini_importance_when_selected=("Gini_Importance", "median"),
        )
    )

    n_completed_folds = int(feature_df["Fold"].nunique())
    n_total_test_predictions = int(shap_df[["Fold", "ModelID"]].drop_duplicates().shape[0])
    summary = shap_summary.merge(feature_summary, on="Gene", how="outer")
    summary["selection_frequency"] = summary["n_selected_folds"] / n_completed_folds
    summary["mean_abs_shap_all_test_predictions"] = (
        summary["sum_abs_shap"] / n_total_test_predictions
    )
    summary["mean_shap_all_test_predictions"] = (
        summary["sum_shap"] / n_total_test_predictions
    )
    summary["n_completed_folds"] = n_completed_folds
    summary["n_total_test_predictions"] = n_total_test_predictions
    return (
        summary.drop(columns=["sum_abs_shap", "sum_shap"])
        .sort_values("mean_abs_shap_all_test_predictions", ascending=False)
        .reset_index(drop=True)
    )


def build_elasticnet_coefficient_summary(coefficient_df, all_genes, completed_folds):
    """Summarize standardized coefficients across completed outer-CV folds."""
    coefficient_matrix = (
        coefficient_df.pivot(
            index="Fold",
            columns="Gene",
            values="Standardized_Coefficient",
        )
        .reindex(index=completed_folds, columns=all_genes)
        .fillna(0.0)
    )
    selected_matrix = (
        coefficient_df.assign(HVG_Selected=1)
        .pivot(index="Fold", columns="Gene", values="HVG_Selected")
        .reindex(index=completed_folds, columns=all_genes)
        .fillna(0.0)
    )

    mean_coefficient = coefficient_matrix.mean(axis=0)
    nonzero_matrix = coefficient_matrix.abs() > 1e-12
    nonzero_count = nonzero_matrix.sum(axis=0)
    direction_matches = coefficient_matrix.apply(
        lambda column: np.sum(
            (np.sign(column) == np.sign(mean_coefficient[column.name]))
            & (column != 0)
        )
    )

    summary = pd.DataFrame(
        {
            "Gene": all_genes,
            "mean_standardized_coefficient": mean_coefficient.to_numpy(),
            "mean_abs_standardized_coefficient": (
                coefficient_matrix.abs().mean(axis=0).to_numpy()
            ),
            "hvg_selection_frequency": selected_matrix.mean(axis=0).to_numpy(),
            "nonzero_frequency": nonzero_matrix.mean(axis=0).to_numpy(),
            "direction_consistency_when_nonzero": np.divide(
                direction_matches.to_numpy(),
                nonzero_count.to_numpy(),
                out=np.zeros(len(all_genes), dtype=float),
                where=nonzero_count.to_numpy() > 0,
            ),
            "n_completed_folds": len(completed_folds),
        }
    )
    summary["abs_mean_standardized_coefficient"] = summary[
        "mean_standardized_coefficient"
    ].abs()
    return (
        summary.loc[summary["hvg_selection_frequency"] > 0]
        .sort_values("abs_mean_standardized_coefficient", ascending=False)
        .reset_index(drop=True)
    )


# =========================
# Main analysis
# =========================
paths = make_case1_paths()
input_file = paths["proc"] / "rna_doubling_time_merged_lung.csv"
fold_results_file = paths["proc"] / "growth_ml_fold_results_lung.csv"
predictions_file = paths["proc"] / "growth_ml_predictions_lung.csv"
rf_importance_file = paths["proc"] / "growth_rf_gene_importance_lung.csv"
rf_fold_features_file = paths["proc"] / "growth_rf_fold_features_lung.csv"
rf_shap_values_file = paths["proc"] / "growth_rf_shap_values_lung.csv"
rf_shap_summary_file = paths["proc"] / "growth_rf_shap_summary_lung.csv"
elasticnet_fold_coefficients_file = (
    paths["proc"] / "growth_elasticnet_fold_coefficients_lung.csv"
)
elasticnet_coefficient_summary_file = (
    paths["proc"] / "growth_elasticnet_coefficient_summary_lung.csv"
)
elasticnet_fold_models_file = paths["proc"] / "growth_elasticnet_fold_models_lung.csv"
lassoed_forest_tree_file = paths["proc"] / "growth_lassoed_forest_tree_summary_lung.csv"
skipped_folds_file = paths["proc"] / "growth_ml_skipped_folds_lung.csv"
config_file = paths["proc"] / "growth_ml_config_lung.json"

df = pd.read_csv(input_file, index_col=0)

target_col = "doubling_time_days"
if target_col not in df.columns:
    raise ValueError(f"Expected target column '{target_col}' in {input_file}")

y_raw = np.log1p(df[target_col].astype(float).values)
X_df_full = df.drop(columns=[target_col])
model_ids = df.index.values

config = {
    "input_file": str(input_file),
    "target_col": target_col,
    "target_transform": "log1p",
    "positive_class": POSITIVE_CLASS_LABEL,
    "n_models": int(df.shape[0]),
    "n_features_input": int(X_df_full.shape[1]),
    "n_hvg_selected_per_fold": N_HVG,
    "fast_quantile": FAST_Q,
    "slow_quantile": SLOW_Q,
    "n_splits": N_SPLITS,
    "n_repeats": N_REPEATS,
    "random_state": RANDOM_STATE,
    "rf_params": RF_PARAMS,
    "rf_shap": {
        "evaluation_samples": "held-out outer-CV extreme samples",
        "explainer": "TreeExplainer",
        "feature_perturbation": "interventional",
        "model_output": "probability",
        "explained_class": POSITIVE_CLASS_LABEL,
        "background": "corresponding outer-fold training extremes",
    },
    "elasticnet_coefficients": {
        "scale": "standardized within each outer-fold training set",
        "absent_feature_contribution": 0,
        "summary_scope": "completed outer-CV folds",
    },
    "elasticnet_l1_ratios": EN_L1_RATIOS,
    "elasticnet_cs": EN_CS,
    "lassoed_forest_cs": LF_CS,
    "logistic_cv_n_jobs": LR_N_JOBS,
    "adaptive_lasso_gamma": ADAPTIVE_LASSO_GAMMA,
    "adaptive_lasso_eps": ADAPTIVE_LASSO_EPS,
}
config_file.write_text(json.dumps(config, indent=2))

print(f"Input matrix: {df.shape[0]} models x {X_df_full.shape[1]} genes")
print(f"HVGs selected inside each training fold: {N_HVG}")
print(f"Positive class: {POSITIVE_CLASS_LABEL} growth (high doubling time)")
print(f"Config saved to: {config_file}")

rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
model_names = ["ElasticNet", "RandomForest", "LassoedForest"]

metrics = {model_name: {"AUROC": [], "Spearman": [], "Pearson": []} for model_name in model_names}
fold_records = []
prediction_records = []
skipped_fold_records = []
rf_importances = []
rf_fold_feature_records = []
rf_shap_records = []
elasticnet_coefficient_records = []
elasticnet_model_records = []
lassoed_forest_tree_records = []

print(f"\nRunning {N_SPLITS * N_REPEATS}-fold repeated CV...")

for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_df_full), start=1):
    X_train_df = X_df_full.iloc[train_idx]
    X_test_df = X_df_full.iloc[test_idx]
    y_train_raw = y_raw[train_idx]
    y_test_raw = y_raw[test_idx]

    fast_cutoff = np.quantile(y_train_raw, FAST_Q)
    slow_cutoff = np.quantile(y_train_raw, SLOW_Q)

    train_extreme_mask = (y_train_raw <= fast_cutoff) | (y_train_raw >= slow_cutoff)
    test_extreme_mask = (y_test_raw <= fast_cutoff) | (y_test_raw >= slow_cutoff)

    X_train_ext_df = X_train_df.loc[train_extreme_mask]
    X_test_ext_df = X_test_df.loc[test_extreme_mask]
    y_train_ext = (y_train_raw[train_extreme_mask] >= slow_cutoff).astype(int)
    y_test_ext = (y_test_raw[test_extreme_mask] >= slow_cutoff).astype(int)
    y_test_raw_ext = y_test_raw[test_extreme_mask]
    test_ids_ext = model_ids[test_idx][test_extreme_mask]

    skip_reason = None
    if X_train_ext_df.shape[0] < 4:
        skip_reason = "too_few_training_extreme_samples"
    elif len(np.unique(y_train_ext)) < 2:
        skip_reason = "training_extremes_have_one_class"
    elif len(np.unique(y_test_ext)) < 2:
        skip_reason = "test_extremes_have_one_class"

    inner_cv = None if skip_reason else safe_inner_cv(y_train_ext, max_splits=4)
    if skip_reason is None and inner_cv is None:
        skip_reason = "too_few_training_samples_for_inner_cv"

    if skip_reason is not None:
        skipped_fold_records.append(
            {
                "Fold": fold_idx,
                "Reason": skip_reason,
                "n_train_extreme": int(X_train_ext_df.shape[0]),
                "n_test_extreme": int(X_test_ext_df.shape[0]),
                "n_train_slow": int(np.sum(y_train_ext)),
                "n_test_slow": int(np.sum(y_test_ext)),
            }
        )
        continue

    hvg = select_hvg_train_only(X_train_ext_df, N_HVG)
    X_train = X_train_ext_df[hvg].values
    X_test = X_test_ext_df[hvg].values
    repeat_idx = ((fold_idx - 1) // N_SPLITS) + 1

    p_rf, rf = fit_predict_random_forest(X_train, y_train_ext, X_test)
    rf_importances.append(pd.Series(rf.feature_importances_, index=hvg, name=f"Fold_{fold_idx}"))
    rf_fold_feature_records.append(
        pd.DataFrame(
            {
                "Fold": fold_idx,
                "Repeat": repeat_idx,
                "Gene": hvg,
                "Gini_Importance": rf.feature_importances_,
            }
        )
    )

    shap_slow, expected_slow = compute_rf_shap(rf, X_train, X_test)
    rf_shap_records.append(
        make_rf_shap_records(
            fold_idx=fold_idx,
            repeat_idx=repeat_idx,
            model_ids=test_ids_ext,
            genes=hvg,
            X_test=X_test,
            y_test=y_test_ext,
            y_test_raw=y_test_raw_ext,
            probabilities=p_rf,
            shap_slow=shap_slow,
            expected_slow=expected_slow,
        )
    )

    p_en, en_model = fit_predict_elasticnet(X_train, y_train_ext, X_test, inner_cv)
    en_coefficients = en_model.coef_.ravel()
    elasticnet_coefficient_records.append(
        pd.DataFrame(
            {
                "Fold": fold_idx,
                "Repeat": repeat_idx,
                "Gene": hvg,
                "Standardized_Coefficient": en_coefficients,
            }
        )
    )
    elasticnet_model_records.append(
        {
            "Fold": fold_idx,
            "Repeat": repeat_idx,
            "C": float(en_model.C_[0]),
            "l1_ratio": float(np.ravel(en_model.l1_ratio_)[0]),
            "n_hvg": len(hvg),
            "n_nonzero": int(np.sum(np.abs(en_coefficients) > 1e-12)),
        }
    )
    p_lf, lf_ridge, lf_lasso, selected_trees = fit_predict_lassoed_forest(
        rf,
        X_train,
        y_train_ext,
        X_test,
        inner_cv,
    )

    model_probabilities = {
        "ElasticNet": p_en,
        "RandomForest": p_rf,
        "LassoedForest": p_lf,
    }

    for model_name, probabilities in model_probabilities.items():
        auroc = roc_auc_score(y_test_ext, probabilities)
        spearman = safe_spearman(probabilities, y_test_raw_ext)
        pearson = safe_pearson(probabilities, y_test_raw_ext)

        metrics[model_name]["AUROC"].append(auroc)
        metrics[model_name]["Spearman"].append(spearman)
        metrics[model_name]["Pearson"].append(pearson)

        fold_records.append(
            {
                "Model": model_name,
                "Fold": fold_idx,
                "AUROC": auroc,
                "Spearman": spearman,
                "Pearson": pearson,
                "n_train_extreme": int(X_train_ext_df.shape[0]),
                "n_test_extreme": int(X_test_ext_df.shape[0]),
                "n_train_fast": int(np.sum(y_train_ext == 0)),
                "n_train_slow": int(np.sum(y_train_ext == 1)),
                "n_test_fast": int(np.sum(y_test_ext == 0)),
                "n_test_slow": int(np.sum(y_test_ext == 1)),
                "fast_cutoff_log1p_dt": fast_cutoff,
                "slow_cutoff_log1p_dt": slow_cutoff,
            }
        )

        for model_id, true_label, raw_dt_log, probability in zip(
            test_ids_ext,
            y_test_ext,
            y_test_raw_ext,
            probabilities,
        ):
            prediction_records.append(
                {
                    "Fold": fold_idx,
                    "Model": model_name,
                    "ModelID": model_id,
                    "True_Label": int(true_label),
                    "Positive_Class": POSITIVE_CLASS_LABEL,
                    "Prob_Slow": float(probability),
                    "log1p_doubling_time_days": float(raw_dt_log),
                }
            )

    lassoed_forest_tree_records.append(
        {
            "Fold": fold_idx,
            "n_trees_total": RF_PARAMS["n_estimators"],
            "n_trees_selected": selected_trees,
            "ridge_C": float(lf_ridge.C_[0]),
            "lasso_C": float(lf_lasso.C_[0]),
        }
    )

    if fold_idx % 5 == 0:
        print(f"  Fold {fold_idx}/{N_SPLITS * N_REPEATS} complete")


results_df = pd.DataFrame(fold_records)
predictions_df = pd.DataFrame(prediction_records)
skipped_folds_df = pd.DataFrame(skipped_fold_records)
if skipped_folds_df.empty:
    skipped_folds_df = pd.DataFrame(columns=SKIPPED_FOLD_COLUMNS)
tree_summary_df = pd.DataFrame(lassoed_forest_tree_records)
rf_fold_features_df = pd.concat(rf_fold_feature_records, ignore_index=True)
rf_shap_values_df = pd.concat(rf_shap_records, ignore_index=True)
rf_shap_summary_df = build_rf_shap_summary(rf_shap_values_df, rf_fold_features_df)
elasticnet_fold_coefficients_df = pd.concat(
    elasticnet_coefficient_records, ignore_index=True
)
elasticnet_fold_models_df = pd.DataFrame(elasticnet_model_records)
elasticnet_coefficient_summary_df = build_elasticnet_coefficient_summary(
    coefficient_df=elasticnet_fold_coefficients_df,
    all_genes=X_df_full.columns,
    completed_folds=elasticnet_fold_models_df["Fold"].tolist(),
)

results_df.to_csv(fold_results_file, index=False)
predictions_df.to_csv(predictions_file, index=False)
skipped_folds_df.to_csv(skipped_folds_file, index=False)
tree_summary_df.to_csv(lassoed_forest_tree_file, index=False)
rf_fold_features_df.to_csv(rf_fold_features_file, index=False)
rf_shap_values_df.to_csv(rf_shap_values_file, index=False)
rf_shap_summary_df.to_csv(rf_shap_summary_file, index=False)
elasticnet_fold_coefficients_df.to_csv(
    elasticnet_fold_coefficients_file, index=False
)
elasticnet_coefficient_summary_df.to_csv(
    elasticnet_coefficient_summary_file, index=False
)
elasticnet_fold_models_df.to_csv(elasticnet_fold_models_file, index=False)

if rf_importances:
    rf_importance_df = (
        pd.concat(rf_importances, axis=1)
        .mean(axis=1)
        .sort_values(ascending=False)
        .rename("mean_gini_importance")
        .to_frame()
    )
    rf_importance_df.to_csv(rf_importance_file)

print("\n===== Growth-classification CV results =====")
for model_name in model_names:
    auroc_mean, auroc_sd = summarize_metric(metrics[model_name]["AUROC"])
    spearman_mean, spearman_sd = summarize_metric(metrics[model_name]["Spearman"])
    pearson_mean, pearson_sd = summarize_metric(metrics[model_name]["Pearson"])

    print(
        f"{model_name:14s} "
        f"AUROC {auroc_mean:.3f} +/- {auroc_sd:.3f} | "
        f"Spearman {spearman_mean:.3f} +/- {spearman_sd:.3f} | "
        f"Pearson {pearson_mean:.3f} +/- {pearson_sd:.3f}"
    )

print(f"\nSaved fold-level results: {fold_results_file}")
print(f"Saved predictions: {predictions_file}")
print(f"Saved skipped-fold log: {skipped_folds_file}")
print(f"Saved Lassoed Forest tree summary: {lassoed_forest_tree_file}")
print(f"Saved RF gene importance: {rf_importance_file}")
print(f"Saved RF fold-level selected features: {rf_fold_features_file}")
print(f"Saved held-out RF SHAP values: {rf_shap_values_file}")
print(f"Saved held-out RF SHAP summary: {rf_shap_summary_file}")
print(f"Saved ElasticNet fold coefficients: {elasticnet_fold_coefficients_file}")
print(f"Saved ElasticNet coefficient summary: {elasticnet_coefficient_summary_file}")
print(f"Saved ElasticNet fold model settings: {elasticnet_fold_models_file}")
