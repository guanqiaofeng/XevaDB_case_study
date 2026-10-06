from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# cat1 (Highly Resistant) ... cat5 (Highly Sensitive), see 4-ml_randomforest.ipynb label_lookup
CAT_MAP = {f"cat{i}": i - 1 for i in range(1, 6)}


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


def make_case2_paths(project_dir=None):
    project_dir = find_project_root() if project_dir is None else Path(project_dir)

    paths = {
        "project": project_dir,
        "raw_pdxe": project_dir / "rawdata" / "pdxe",
        "raw_expert_rating": project_dir / "rawdata" / "expert_rating",
        "datasets": project_dir / "procdata" / "pdxe",
        "proc": project_dir / "results" / "2_drugSensitivity",
        "results": project_dir / "results" / "2_drugSensitivity",
        "figures": project_dir / "figures_tables" / "figures_main",
        "figures_supp": project_dir / "figures_tables" / "figures_supp",
        "tables": project_dir / "figures_tables" / "tables",
    }

    paths["proc"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["figures_supp"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)

    return paths


def load_expert_rating_labels(expert_rating_table):
    """Load consolidated expert ratings, drop rows with no rater majority, map cat1..cat5 to 0..4."""
    df = pd.read_csv(expert_rating_table)
    df_clean = df[~df["rating_merge"].str.contains("/", na=False)].copy()
    df_clean["label"] = df_clean["rating_merge"].map(CAT_MAP)
    return df_clean


def make_case2_cv_folds(df_clean, n_splits=5, random_state=42):
    """Stratified k-fold assignment on `label`, shared by every model trained in case 2.

    Every figure is held out as test exactly once, across `n_splits` folds; the remaining
    folds form that iteration's training pool. Each model carves its own validation signal
    out of that pool (RandomForest via its own inner CV, ResNet18 via a manually carved-out
    validation slice for early stopping) -- the held-out fold itself is only ever scored once.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_df = df_clean.copy()
    fold_df["fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(skf.split(fold_df, fold_df["label"])):
        fold_df.iloc[test_idx, fold_df.columns.get_loc("fold")] = fold_idx

    fold_df = fold_df.sort_values("figure_id").reset_index(drop=True)
    return fold_df
