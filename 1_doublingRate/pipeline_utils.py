import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Doubling-time estimation parameters
MAX_DAY = 35       # use first 35 days to estimate exponential growth
MIN_POINTS = 3     # minimum number of time points per mouse
MIN_R2 = 0.8       # minimum log-linear fit quality

# machine learning parameters
RANDOM_STATE = 42


def find_project_root(start=None, marker_dirs=("rawdata", "procdata")):
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()

    for path in [start, *start.parents]:
        if all((path / marker).exists() for marker in marker_dirs):
            return path

    raise FileNotFoundError(
        f"Could not find project root from {start}. "
        f"Expected markers: {marker_dirs}"
    )


def make_case1_paths(project_dir=None):
    project_dir = find_project_root() if project_dir is None else Path(project_dir)

    paths = {
        "project": project_dir,
        "raw_uhn": project_dir / "procdata",
        "proc": project_dir / "results" / "1_doublingRate",
        "results": project_dir / "results" / "1_doublingRate",
        "figures": project_dir / "figures_tables" / "figures_main",
    }

    paths["proc"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    return paths

def prepare_mouse_fit_data(control_df, mouse_id, max_day=MAX_DAY):
    fit_df = (
        control_df.loc[
            (control_df["model.id"] == mouse_id) & (control_df["time"] <= max_day),
            ["time", "volume"]
        ]
        .dropna()
        .sort_values("time")
    )

    fit_df = (
        fit_df.groupby("time", as_index=False)["volume"]
        .mean()
        .sort_values("time")
    )

    return fit_df


def plot_mouse_growth_fit(mouse_id, ax_raw, ax_log, control_df, max_day=MAX_DAY):
    fit_df = prepare_mouse_fit_data(control_df, mouse_id, max_day=max_day)

    if fit_df.empty:
        ax_raw.set_title(f"No data: {mouse_id}")
        ax_log.set_axis_off()
        return

    ax_raw.plot(
        fit_df["time"],
        fit_df["volume"],
        "o-",
        color="tab:blue",
        markersize=4,
    )
    ax_raw.set_title(f"Raw: {mouse_id}")
    ax_raw.set_ylabel("Volume (mm³)")

    if fit_df.shape[0] < MIN_POINTS:
        ax_log.set_title("Too few points")
        ax_log.set_axis_off()
        return

    x = fit_df["time"]
    y = np.log(fit_df["volume"])

    slope, intercept, r, p_value, std_err = linregress(x, y)
    r2 = r**2
    doubling_time = np.log(2) / slope if slope > 0 else np.nan

    ax_log.scatter(x, y, color="black", s=20)
    ax_log.plot(x, intercept + slope * x, "r--", alpha=0.8)

    ax_log.set_title(f"Log-linear fit: R²={r2:.2f} | DT={doubling_time:.1f} d")
    ax_log.set_xlabel("Time (days)")
    ax_log.set_ylabel("ln(Volume)")

# Machine-learning helper functions

def select_hvg_train_only(X_train_df, n_hvg):
    return X_train_df.var(axis=0).sort_values(ascending=False).head(n_hvg).index.tolist()


def safe_spearman(a, b):
    r = spearmanr(a, b).correlation
    return float(0.0 if np.isnan(r) else r)


def safe_pearson(a, b):
    r = pearsonr(a, b)[0]
    return float(0.0 if np.isnan(r) else r)


def safe_inner_cv(y):
    """Inner CV for hyperparameter selection: a flat 4-fold stratified split.

    min(4, minimum per-class training count) was checked directly against the
    actual data (breast: always 20 per class; lung: always 6) and always
    resolves to 4, so the adaptive cap is stated flatly here instead. The
    degenerate-fold guard is kept (not just inlined away) so fold-exclusion
    criterion (iv) -- too few training-extreme samples per class for a valid
    inner split -- stays a real, reachable check rather than dead code, in
    case a future data revision ever drops a fold's per-class count below 4.
    """
    class_counts = np.bincount(np.asarray(y, dtype=int))
    min_class_count = class_counts[class_counts > 0].min()
    if min_class_count < 4:
        return None

    return StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def per_tree_probability_matrix(rf_model, X, positive_class=1):
    """Represent each sample by the positive-class probability from each RF tree."""
    tree_probabilities = []

    for tree in rf_model.estimators_:
        proba = tree.predict_proba(X)
        classes = list(tree.classes_)

        if positive_class in classes:
            positive_index = classes.index(positive_class)
            tree_probabilities.append(proba[:, positive_index])
        else:
            tree_probabilities.append(np.zeros(X.shape[0]))

    return np.column_stack(tree_probabilities)


# =========================
# Growth-classification ML pipeline (shared across cohorts)
# =========================

GROWTH_ML_SKIPPED_FOLD_COLUMNS = [
    "Fold",
    "Reason",
    "n_train_extreme",
    "n_test_extreme",
    "n_train_slow",
    "n_test_slow",
]


def fit_predict_elasticnet(X_train, y_train, X_test, inner_cv, *, l1_ratios, cs, random_state, n_jobs):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=l1_ratios,
        Cs=cs,
        cv=inner_cv,
        max_iter=20000,
        n_jobs=n_jobs,
        random_state=random_state,
        scoring="roc_auc",
    )
    model.fit(X_train_scaled, y_train)

    return model.predict_proba(X_test_scaled)[:, 1], model


def fit_predict_random_forest(X_train, y_train, X_test, rf_params):
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)

    return model.predict_proba(X_test)[:, 1], model


def fit_predict_lassoed_forest(
    rf_model, X_train, y_train, X_test, inner_cv, *,
    cs, random_state, n_jobs, adaptive_lasso_gamma, adaptive_lasso_eps,
):
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
        Cs=cs,
        cv=inner_cv,
        max_iter=20000,
        n_jobs=n_jobs,
        random_state=random_state,
        scoring="roc_auc",
    )
    ridge.fit(Z_train_scaled, y_train)

    initial_beta = ridge.coef_.ravel()
    penalty_weights = 1.0 / (np.abs(initial_beta) + adaptive_lasso_eps) ** adaptive_lasso_gamma

    Z_train_weighted = Z_train_scaled / penalty_weights
    Z_test_weighted = Z_test_scaled / penalty_weights

    lasso = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        Cs=cs,
        cv=inner_cv,
        max_iter=20000,
        n_jobs=n_jobs,
        random_state=random_state,
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


def build_rf_param_grid(max_depth_grid, min_samples_leaf_grid, *, n_estimators, max_features, random_state):
    return [
        dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        for max_depth in max_depth_grid
        for min_samples_leaf in min_samples_leaf_grid
    ]


def select_random_forest_hyperparameters(X_train, y_train, inner_cv, param_grid):
    """Nested selection: pick the RF hyperparameter combination with the best
    mean inner-CV AUROC on `X_train`/`y_train`, using sklearn's GridSearchCV
    over inner_cv's own splits -- never the outer test fold. Matches case
    study 4's grid-search implementation (previously hand-rolled here);
    verified to select identical hyperparameters on real data before
    switching, and inner_cv's folds are never degenerate in practice (checked
    directly), so GridSearchCV's default error handling is never exercised.

    `param_grid` is the list-of-full-parameter-dicts produced by
    `build_rf_param_grid` (only max_depth/min_samples_leaf actually vary
    across entries); the shared base kwargs (n_estimators, max_features,
    random_state, n_jobs) are read from the first entry to build the base
    estimator GridSearchCV tunes on top of.
    """
    base_kwargs = {k: v for k, v in param_grid[0].items() if k not in ("max_depth", "min_samples_leaf")}
    base_estimator = RandomForestClassifier(**base_kwargs)
    sklearn_grid = {
        "max_depth": sorted({p["max_depth"] for p in param_grid}),
        "min_samples_leaf": sorted({p["min_samples_leaf"] for p in param_grid}),
    }

    search = GridSearchCV(base_estimator, param_grid=sklearn_grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
    search.fit(X_train, y_train)

    best_params = {**base_kwargs, **search.best_params_}
    return best_params, search.best_score_


def select_lassoed_forest_hyperparameters(
    X_train, y_train, inner_cv, param_grid, *,
    scoring_c, random_state, adaptive_lasso_gamma, adaptive_lasso_eps,
):
    """Nested selection for LassoedForest's own RF hyperparameters, scored by
    the *post-lasso* AUROC (not the raw RF AUROC) on inner_cv's own splits.

    The ridge/lasso step normally gets its own inner-CV-selected C (see
    `fit_predict_lassoed_forest`); repeating that full search inside every
    grid point of *this* search would need a third level of nested CV on
    folds that are already down to a handful of samples, so this scoring
    pass uses a single fixed `scoring_c` for both ridge and lasso instead.
    Once the winning max_depth/min_samples_leaf is chosen, the final model
    (fit by the caller) still gets the full, properly nested ridge/lasso C
    search via `fit_predict_lassoed_forest`.
    """
    best_params, best_score = param_grid[0], -np.inf
    for rf_params in param_grid:
        scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
            y_tr, y_val = y_train[inner_train_idx], y_train[inner_val_idx]
            if len(np.unique(y_val)) < 2 or len(np.unique(y_tr)) < 2:
                continue

            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train[inner_train_idx], y_tr)

            Z_tr = per_tree_probability_matrix(rf, X_train[inner_train_idx])
            Z_val = per_tree_probability_matrix(rf, X_train[inner_val_idx])
            scaler = StandardScaler()
            Z_tr_scaled = scaler.fit_transform(Z_tr)
            Z_val_scaled = scaler.transform(Z_val)

            ridge = LogisticRegression(
                penalty="l2", solver="lbfgs", C=scoring_c,
                max_iter=20000, random_state=random_state,
            )
            ridge.fit(Z_tr_scaled, y_tr)
            weights = 1.0 / (np.abs(ridge.coef_.ravel()) + adaptive_lasso_eps) ** adaptive_lasso_gamma

            lasso = LogisticRegression(
                penalty="l1", solver="saga", C=scoring_c,
                max_iter=20000, random_state=random_state,
            )
            lasso.fit(Z_tr_scaled / weights, y_tr)
            proba = lasso.predict_proba(Z_val_scaled / weights)[:, 1]
            scores.append(roc_auc_score(y_val, proba))

        if scores and np.mean(scores) > best_score:
            best_score = float(np.mean(scores))
            best_params = rf_params
    return best_params, best_score


def run_growth_ml_pipeline(
    cohort,
    paths,
    *,
    n_splits,
    n_hvg,
    rf_max_depth_grid=(3, 6),
    rf_min_samples_leaf_grid=(1, 2),
    rf_n_estimators=500,
    rf_max_features="sqrt",
    n_repeats=10,
    random_state=RANDOM_STATE,
    fast_q=0.33,
    slow_q=0.67,
    positive_class_label="slow",
    en_l1_ratios=(0.1, 0.5, 0.9),
    en_cs=(0.01, 0.1, 1.0, 10.0),
    lf_cs=(0.001, 0.01, 0.1, 1.0, 10.0, 100),
    lf_scoring_c=1.0,
    lr_n_jobs=1,
    adaptive_lasso_gamma=1.0,
    adaptive_lasso_eps=1e-3,
):
    """Run the ElasticNet / RandomForest / LassoedForest growth-classification
    repeated-CV pipeline for one cohort and write all fold-level, prediction,
    and summary tables under paths["proc"].

    `cohort` (e.g. "breast", "lung") is used both to locate the merged
    RNA + doubling-time input table and to name every output file, so the
    two driver scripts differ only in the config values they pass in here.

    RandomForest and LassoedForest each get their own independent nested
    hyperparameter search over `rf_max_depth_grid x rf_min_samples_leaf_grid`,
    scored on inner_cv splits of that fold's training-extreme data -- RandomForest
    on its own raw AUROC, LassoedForest on its own post-lasso AUROC. `n_hvg` stays
    fixed (not searched) and shared across all three models, per cohort.
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")

    input_file = paths["proc"] / f"rna_doubling_time_merged_{cohort}.csv"
    fold_results_file = paths["proc"] / f"growth_ml_fold_results_{cohort}.csv"
    predictions_file = paths["proc"] / f"growth_ml_predictions_{cohort}.csv"
    rf_importance_file = paths["proc"] / f"growth_rf_gene_importance_{cohort}.csv"
    rf_fold_features_file = paths["proc"] / f"growth_rf_fold_features_{cohort}.csv"
    rf_hyperparameter_selection_file = paths["proc"] / f"growth_rf_hyperparameter_selection_{cohort}.csv"
    rf_shap_values_file = paths["proc"] / f"growth_rf_shap_values_{cohort}.csv"
    rf_shap_summary_file = paths["proc"] / f"growth_rf_shap_summary_{cohort}.csv"
    elasticnet_fold_coefficients_file = (
        paths["proc"] / f"growth_elasticnet_fold_coefficients_{cohort}.csv"
    )
    elasticnet_coefficient_summary_file = (
        paths["proc"] / f"growth_elasticnet_coefficient_summary_{cohort}.csv"
    )
    elasticnet_fold_models_file = paths["proc"] / f"growth_elasticnet_fold_models_{cohort}.csv"
    lassoed_forest_tree_file = paths["proc"] / f"growth_lassoed_forest_tree_summary_{cohort}.csv"
    skipped_folds_file = paths["proc"] / f"growth_ml_skipped_folds_{cohort}.csv"
    config_file = paths["proc"] / f"growth_ml_config_{cohort}.json"

    # Sorted so CV fold assignment (RepeatedKFold shuffles by row position)
    # is independent of whatever column order the upstream omics extraction
    # happens to produce.
    df = pd.read_csv(input_file, index_col=0).sort_index()

    target_col = "doubling_time_days"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in {input_file}")

    y_raw = np.log1p(df[target_col].astype(float).values)
    X_df_full = df.drop(columns=[target_col])
    model_ids = df.index.values

    config = {
        "cohort": cohort,
        "input_file": str(input_file),
        "target_col": target_col,
        "target_transform": "log1p",
        "positive_class": positive_class_label,
        "n_models": int(df.shape[0]),
        "n_features_input": int(X_df_full.shape[1]),
        "n_hvg_selected_per_fold": n_hvg,
        "fast_quantile": fast_q,
        "slow_quantile": slow_q,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "rf_n_estimators": rf_n_estimators,
        "rf_max_features": rf_max_features,
        "rf_max_depth_grid": list(rf_max_depth_grid),
        "rf_min_samples_leaf_grid": list(rf_min_samples_leaf_grid),
        "rf_hyperparameter_selection": (
            "nested per outer fold via inner_cv; RandomForest selects on its own "
            "raw AUROC, LassoedForest selects independently on its own post-lasso AUROC"
        ),
        "lf_scoring_c": lf_scoring_c,
        "rf_shap": {
            "evaluation_samples": "held-out outer-CV extreme samples",
            "explainer": "TreeExplainer",
            "feature_perturbation": "interventional",
            "model_output": "probability",
            "explained_class": positive_class_label,
            "background": "corresponding outer-fold training extremes",
        },
        "elasticnet_coefficients": {
            "scale": "standardized within each outer-fold training set",
            "absent_feature_contribution": 0,
            "summary_scope": "completed outer-CV folds",
        },
        "elasticnet_l1_ratios": list(en_l1_ratios),
        "elasticnet_cs": list(en_cs),
        "lassoed_forest_cs": list(lf_cs),
        "logistic_cv_n_jobs": lr_n_jobs,
        "adaptive_lasso_gamma": adaptive_lasso_gamma,
        "adaptive_lasso_eps": adaptive_lasso_eps,
    }
    config_file.write_text(json.dumps(config, indent=2))

    print(f"[{cohort}] Input matrix: {df.shape[0]} models x {X_df_full.shape[1]} genes")
    print(f"[{cohort}] HVGs selected inside each training fold: {n_hvg}")
    print(f"[{cohort}] Positive class: {positive_class_label} growth (high doubling time)")
    print(f"[{cohort}] Config saved to: {config_file}")

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    model_names = ["ElasticNet", "RandomForest", "LassoedForest"]

    rf_param_grid = build_rf_param_grid(
        rf_max_depth_grid, rf_min_samples_leaf_grid,
        n_estimators=rf_n_estimators, max_features=rf_max_features, random_state=random_state,
    )

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
    rf_hyperparameter_records = []

    print(f"\n[{cohort}] Running {n_splits * n_repeats}-fold repeated CV...")

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_df_full), start=1):
        X_train_df = X_df_full.iloc[train_idx]
        X_test_df = X_df_full.iloc[test_idx]
        y_train_raw = y_raw[train_idx]
        y_test_raw = y_raw[test_idx]

        fast_cutoff = np.quantile(y_train_raw, fast_q)
        slow_cutoff = np.quantile(y_train_raw, slow_q)

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

        inner_cv = None if skip_reason else safe_inner_cv(y_train_ext)
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

        hvg = select_hvg_train_only(X_train_ext_df, n_hvg)
        X_train = X_train_ext_df[hvg].values
        X_test = X_test_ext_df[hvg].values
        repeat_idx = ((fold_idx - 1) // n_splits) + 1

        rf_selected_params, rf_inner_score = select_random_forest_hyperparameters(
            X_train, y_train_ext, inner_cv, rf_param_grid,
        )
        p_rf, rf = fit_predict_random_forest(X_train, y_train_ext, X_test, rf_selected_params)
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

        p_en, en_model = fit_predict_elasticnet(
            X_train, y_train_ext, X_test, inner_cv,
            l1_ratios=en_l1_ratios, cs=en_cs, random_state=random_state, n_jobs=lr_n_jobs,
        )
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
        lf_selected_params, lf_inner_score = select_lassoed_forest_hyperparameters(
            X_train, y_train_ext, inner_cv, rf_param_grid,
            scoring_c=lf_scoring_c, random_state=random_state,
            adaptive_lasso_gamma=adaptive_lasso_gamma, adaptive_lasso_eps=adaptive_lasso_eps,
        )
        lf_rf = RandomForestClassifier(**lf_selected_params)
        lf_rf.fit(X_train, y_train_ext)
        p_lf, lf_ridge, lf_lasso, selected_trees = fit_predict_lassoed_forest(
            lf_rf, X_train, y_train_ext, X_test, inner_cv,
            cs=lf_cs, random_state=random_state, n_jobs=lr_n_jobs,
            adaptive_lasso_gamma=adaptive_lasso_gamma, adaptive_lasso_eps=adaptive_lasso_eps,
        )

        rf_hyperparameter_records.append(
            {
                "Fold": fold_idx,
                "Repeat": repeat_idx,
                "Model": "RandomForest",
                "selected_max_depth": rf_selected_params["max_depth"],
                "selected_min_samples_leaf": rf_selected_params["min_samples_leaf"],
                "inner_cv_auroc": rf_inner_score,
            }
        )
        rf_hyperparameter_records.append(
            {
                "Fold": fold_idx,
                "Repeat": repeat_idx,
                "Model": "LassoedForest",
                "selected_max_depth": lf_selected_params["max_depth"],
                "selected_min_samples_leaf": lf_selected_params["min_samples_leaf"],
                "inner_cv_auroc": lf_inner_score,
            }
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
                        "Positive_Class": positive_class_label,
                        "Prob_Slow": float(probability),
                        "log1p_doubling_time_days": float(raw_dt_log),
                    }
                )

        lassoed_forest_tree_records.append(
            {
                "Fold": fold_idx,
                "n_trees_total": rf_n_estimators,
                "n_trees_selected": selected_trees,
                "selected_max_depth": lf_selected_params["max_depth"],
                "selected_min_samples_leaf": lf_selected_params["min_samples_leaf"],
                "ridge_C": float(lf_ridge.C_[0]),
                "lasso_C": float(lf_lasso.C_[0]),
            }
        )

        if fold_idx % 5 == 0:
            print(f"[{cohort}]   Fold {fold_idx}/{n_splits * n_repeats} complete")

    results_df = pd.DataFrame(fold_records)
    predictions_df = pd.DataFrame(prediction_records)
    skipped_folds_df = pd.DataFrame(skipped_fold_records)
    if skipped_folds_df.empty:
        skipped_folds_df = pd.DataFrame(columns=GROWTH_ML_SKIPPED_FOLD_COLUMNS)
    tree_summary_df = pd.DataFrame(lassoed_forest_tree_records)
    rf_hyperparameter_df = pd.DataFrame(rf_hyperparameter_records)
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
    rf_hyperparameter_df.to_csv(rf_hyperparameter_selection_file, index=False)
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

    print(f"\n[{cohort}] ===== Growth-classification CV results =====")
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

    print(f"\n[{cohort}] Saved fold-level results: {fold_results_file}")
    print(f"[{cohort}] Saved predictions: {predictions_file}")
    print(f"[{cohort}] Saved skipped-fold log: {skipped_folds_file}")
    print(f"[{cohort}] Saved Lassoed Forest tree summary: {lassoed_forest_tree_file}")
    print(f"[{cohort}] Saved RF/LassoedForest hyperparameter selection: {rf_hyperparameter_selection_file}")
    print(f"[{cohort}] Saved RF gene importance: {rf_importance_file}")
    print(f"[{cohort}] Saved RF fold-level selected features: {rf_fold_features_file}")
    print(f"[{cohort}] Saved held-out RF SHAP values: {rf_shap_values_file}")
    print(f"[{cohort}] Saved held-out RF SHAP summary: {rf_shap_summary_file}")
    print(f"[{cohort}] Saved ElasticNet fold coefficients: {elasticnet_fold_coefficients_file}")
    print(f"[{cohort}] Saved ElasticNet coefficient summary: {elasticnet_coefficient_summary_file}")
    print(f"[{cohort}] Saved ElasticNet fold model settings: {elasticnet_fold_models_file}")


def compare_hvg_across_models(
    df, *, n_hvg_grid, n_splits, n_repeats, rf_params,
    random_state=RANDOM_STATE, fast_q=0.33, slow_q=0.67,
    en_l1_ratios=(0.1, 0.5, 0.9), en_cs=(0.01, 0.1, 1.0, 10.0), lr_n_jobs=1,
):
    """Compare ElasticNet and RandomForest AUROC/Spearman across candidate N_HVG
    values, holding every other hyperparameter fixed, to check whether a cohort's
    N_HVG preference -- originally found via LassoedForest's own grid search,
    saved as `growth_lassoedforest_sensitivity_summary_{cohort}.csv` -- generalizes
    to the other two models, or is specific to LassoedForest's tree-probability
    architecture. See `supp/hvg_selection_justification.ipynb`.
    """
    y_raw = np.log1p(df["doubling_time_days"].astype(float).values)
    X_df_full = df.drop(columns=["doubling_time_days"])

    rows = []
    for n_hvg in n_hvg_grid:
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        metrics = {m: {"AUROC": [], "Spearman": []} for m in ["ElasticNet", "RandomForest"]}
        n_completed = 0

        for train_idx, test_idx in rkf.split(X_df_full):
            X_train_df, X_test_df = X_df_full.iloc[train_idx], X_df_full.iloc[test_idx]
            y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

            fast_cutoff, slow_cutoff = np.quantile(y_train_raw, fast_q), np.quantile(y_train_raw, slow_q)
            train_ext = (y_train_raw <= fast_cutoff) | (y_train_raw >= slow_cutoff)
            test_ext = (y_test_raw <= fast_cutoff) | (y_test_raw >= slow_cutoff)

            X_train_ext_df, X_test_ext_df = X_train_df.loc[train_ext], X_test_df.loc[test_ext]
            y_train_ext = (y_train_raw[train_ext] >= slow_cutoff).astype(int)
            y_test_ext = (y_test_raw[test_ext] >= slow_cutoff).astype(int)
            y_test_raw_ext = y_test_raw[test_ext]

            if X_train_ext_df.shape[0] < 4 or len(np.unique(y_train_ext)) < 2 or len(np.unique(y_test_ext)) < 2:
                continue
            inner_cv = safe_inner_cv(y_train_ext)
            if inner_cv is None:
                continue

            hvg = select_hvg_train_only(X_train_ext_df, n_hvg)
            X_train, X_test = X_train_ext_df[hvg].values, X_test_ext_df[hvg].values
            n_completed += 1

            p_rf, _ = fit_predict_random_forest(X_train, y_train_ext, X_test, rf_params)
            p_en, _ = fit_predict_elasticnet(
                X_train, y_train_ext, X_test, inner_cv,
                l1_ratios=en_l1_ratios, cs=en_cs, random_state=random_state, n_jobs=lr_n_jobs,
            )
            for model_name, probabilities in [("ElasticNet", p_en), ("RandomForest", p_rf)]:
                metrics[model_name]["AUROC"].append(roc_auc_score(y_test_ext, probabilities))
                metrics[model_name]["Spearman"].append(safe_spearman(probabilities, y_test_raw_ext))

        for model_name in ["ElasticNet", "RandomForest"]:
            auroc_mean, auroc_sd = summarize_metric(metrics[model_name]["AUROC"])
            spearman_mean, spearman_sd = summarize_metric(metrics[model_name]["Spearman"])
            rows.append(
                {
                    "n_hvg": n_hvg,
                    "Model": model_name,
                    "n_completed_folds": n_completed,
                    "mean_AUROC": auroc_mean,
                    "sd_AUROC": auroc_sd,
                    "mean_Spearman": spearman_mean,
                    "sd_Spearman": spearman_sd,
                }
            )

    return pd.DataFrame(rows)


# =========================
# DT-fit QC threshold sensitivity (shared across cohorts)
# =========================


def fit_all_mouse_growth_curves(control_df, max_day, *, min_points=2, modelid_split_index=0):
    """Fit an early-phase log-linear growth model for every control-arm mouse.

    Unlike the QC loop in the 1-*_DT_preprocessing notebooks, this keeps a
    row for every mouse with enough points for *some* fit (`min_points`,
    default 2 -- the minimum linregress needs), without applying the
    pipeline's MIN_POINTS/MIN_R2 thresholds. That lets `qc_threshold_sensitivity`
    re-apply different thresholds to the same fits without refitting.
    """
    records = []
    for mouse_id, mouse_df in control_df.groupby("model.id"):
        fit_df = (
            mouse_df.loc[mouse_df["time"] <= max_day, ["time", "volume"]]
            .dropna()
            .sort_values("time")
        )
        fit_df = (
            fit_df.groupby("time", as_index=False)["volume"]
            .mean()
            .sort_values("time")
        )

        n_points = fit_df.shape[0]
        slope, r2 = np.nan, np.nan
        if n_points >= min_points:
            slope, _, r, _, _ = linregress(fit_df["time"], np.log(fit_df["volume"]))
            r2 = r ** 2

        records.append(
            {
                "mouse_id": mouse_id,
                "modelID_raw": mouse_id.split(".")[modelid_split_index],
                "n_points": n_points,
                "growth_rate_k": slope,
                "r2": r2,
            }
        )

    return pd.DataFrame(records)


def qc_threshold_sensitivity(
    control_df,
    *,
    baseline_max_day,
    baseline_min_points,
    baseline_min_r2,
    min_points_grid,
    min_r2_grid,
    modelid_split_index=0,
):
    """One-parameter-at-a-time sensitivity sweep over MIN_POINTS and MIN_R2,
    holding MAX_DAY fixed at `baseline_max_day` and the other threshold at its
    pipeline-default (baseline) value.

    MAX_DAY itself is not swept here: in this dataset, follow-up duration is
    confounded with growth rate (see `plot_followup_censoring`), so a
    "retained count vs. MAX_DAY" sweep would be misleading -- widening the
    window doesn't add unbiased data, it disproportionately adds late-phase
    data for slow-growing tumors that haven't yet hit the ethical size limit.
    """

    def summarize(fit_df, min_points, min_r2):
        passing = fit_df.loc[
            (fit_df["n_points"] >= min_points)
            & (fit_df["growth_rate_k"] > 0)
            & (fit_df["r2"] >= min_r2)
        ]
        doubling_time = np.log(2) / passing["growth_rate_k"]
        return {
            "n_mice_retained": int(passing.shape[0]),
            "n_models_retained": int(passing["modelID_raw"].nunique()),
            "median_doubling_time_days": (
                float(doubling_time.median()) if not passing.empty else np.nan
            ),
            "mean_r2": float(passing["r2"].mean()) if not passing.empty else np.nan,
        }

    rows = []

    baseline_fit_df = fit_all_mouse_growth_curves(
        control_df, max_day=baseline_max_day, modelid_split_index=modelid_split_index
    )

    for min_points in min_points_grid:
        rows.append(
            {
                "parameter": "MIN_POINTS",
                "value": min_points,
                "is_baseline": min_points == baseline_min_points,
                **summarize(baseline_fit_df, min_points, baseline_min_r2),
            }
        )

    for min_r2 in min_r2_grid:
        rows.append(
            {
                "parameter": "MIN_R2",
                "value": min_r2,
                "is_baseline": min_r2 == baseline_min_r2,
                **summarize(baseline_fit_df, baseline_min_points, min_r2),
            }
        )

    return pd.DataFrame(rows)


def plot_qc_threshold_sensitivity(sensitivity_df, out_path, cohort_label):
    """2-panel figure: retained mouse/model counts vs MIN_POINTS and MIN_R2,
    one parameter per panel, with the pipeline's baseline value marked."""
    params = ["MIN_POINTS", "MIN_R2"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))

    for ax, param in zip(axes, params):
        sub = sensitivity_df.loc[sensitivity_df["parameter"] == param].sort_values("value")
        ax.plot(sub["value"], sub["n_mice_retained"], "o-", color="tab:blue", label="Mice retained")
        ax.plot(sub["value"], sub["n_models_retained"], "s--", color="tab:orange", label="Models retained")

        baseline_value = sub.loc[sub["is_baseline"], "value"]
        if not baseline_value.empty:
            ax.axvline(baseline_value.iloc[0], color="black", linestyle=":", linewidth=1)

        ax.set_xlabel(param)
        ax.set_ylabel("N retained")
        ax.set_title(param)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"{cohort_label}: DT-fit QC threshold sensitivity", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def compute_followup_censoring(control_df, per_mouse_df):
    """Merge each mouse's raw (untruncated by MAX_DAY) last-observed day with
    its baseline-fit growth rate, to check whether follow-up duration is
    confounded with growth rate (i.e. ethical-endpoint censoring)."""
    last_day = (
        control_df.groupby("model.id")["time"]
        .max()
        .rename("last_observed_day")
        .reset_index()
        .rename(columns={"model.id": "mouse_id"})
    )
    return per_mouse_df.merge(last_day, on="mouse_id", how="inner")


def compute_maxday_composition_drift(
    control_df, max_day_grid, *, min_points=3, min_r2=0.8, modelid_split_index=0
):
    """For each candidate MAX_DAY, refit every mouse and summarize the
    doubling-time distribution of the retained sample.

    A stable median across max_day_grid would indicate the retained sample's
    composition isn't sensitive to the window length; a steady drift is the
    direct signature of censoring-driven composition bias -- slow-growing
    tumors keep gaining data as the window widens while fast-growing tumors
    are already capped by their own natural (ethical-endpoint) censoring.
    """
    rows = []
    for max_day in max_day_grid:
        fit_df = fit_all_mouse_growth_curves(
            control_df, max_day=max_day, modelid_split_index=modelid_split_index
        )
        kept = fit_df.loc[
            (fit_df["n_points"] >= min_points)
            & (fit_df["growth_rate_k"] > 0)
            & (fit_df["r2"] >= min_r2)
        ]
        doubling_time = np.log(2) / kept["growth_rate_k"]
        rows.append(
            {
                "MAX_DAY": max_day,
                "n_retained": int(kept.shape[0]),
                "median_growth_rate_k": (
                    float(kept["growth_rate_k"].median()) if not kept.empty else np.nan
                ),
                "median_doubling_time_days": (
                    float(doubling_time.median()) if not kept.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_followup_censoring(censoring_df, drift_df, max_day, out_path, cohort_label):
    """3-panel figure motivating a bounded early-phase fit window:

    (1) distribution of each mouse's raw last-observed day, with `max_day`
        marked, showing how much real follow-up exists beyond the cutoff;
    (2) last-observed day vs. fitted growth rate, showing that follow-up
        duration is itself confounded with growth rate -- fast-growing
        tumors tend to be followed for fewer days (ethical size-limit
        sacrifice) -- so widening the fit window is not a neutral choice;
    (3) the retained sample's median doubling time as a function of
        MAX_DAY itself -- a continuous drift (rather than a plateau) is the
        signature of that same censoring bias accumulating as the window
        widens, and shows there is no data-driven "safe" cutoff to find.
    """
    rho, p_value = spearmanr(censoring_df["last_observed_day"], censoring_df["growth_rate_k"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ax.hist(censoring_df["last_observed_day"], bins=30, color="tab:blue", edgecolor="black", alpha=0.85)
    ax.axvline(max_day, color="black", linestyle=":", linewidth=1.3)
    frac_beyond = (censoring_df["last_observed_day"] > max_day).mean()
    ax.text(
        0.97, 0.95, f"{frac_beyond:.0%} of mice followed\nbeyond day {max_day}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )
    ax.set_xlabel("Last observed day (uncensored by MAX_DAY)")
    ax.set_ylabel("N mice")
    ax.set_title("Raw follow-up duration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.scatter(
        censoring_df["last_observed_day"], censoring_df["growth_rate_k"],
        s=18, alpha=0.6, color="tab:orange", edgecolor="black", linewidth=0.3,
    )
    ax.axvline(max_day, color="black", linestyle=":", linewidth=1.3)
    ax.set_xlabel("Last observed day (uncensored by MAX_DAY)")
    ax.set_ylabel("Fitted growth rate k (baseline fit)")
    ax.set_title(f"Spearman r = {rho:.2f} (p = {p_value:.1e})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    drift_sorted = drift_df.sort_values("MAX_DAY")
    ax.plot(
        drift_sorted["MAX_DAY"], drift_sorted["median_doubling_time_days"],
        "o-", color="tab:green",
    )
    ax.axvline(max_day, color="black", linestyle=":", linewidth=1.3)
    ax.set_xlabel("Candidate MAX_DAY")
    ax.set_ylabel("Median retained DT (days)")
    ax.set_title("Retained-sample composition drift")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"{cohort_label}: follow-up duration is confounded with growth rate", y=1.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig
