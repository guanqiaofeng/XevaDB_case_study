from pathlib import Path
from scipy.stats import linregress
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold

# Doubling-time estimation parameters
MAX_DAY = 35       # use first 35 days to estimate exponential growth
MIN_POINTS = 3     # minimum number of time points per mouse
MIN_R2 = 0.8       # minimum log-linear fit quality

# machine learning parameters
RANDOM_STATE = 42


def find_project_root(start=None, marker_dirs=("data", "1_doublingRate")):
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
        "data": project_dir / "data",
        "raw_uhn": project_dir / "data" / "procdata" / "0_datasets",
        "proc": project_dir / "data" / "procdata" / "1_doublingRate",
        "results": project_dir / "results" / "1_doublingRate",
        "figures": project_dir / "figures",
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


def safe_inner_cv(y, max_splits=4):
    class_counts = np.bincount(np.asarray(y, dtype=int))
    min_class_count = class_counts[class_counts > 0].min()
    n_splits = min(max_splits, int(min_class_count))
    if n_splits < 2:
        return None

    return StratifiedKFold(
        n_splits=n_splits,
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
