"""Reusable feature-set helpers for the paclitaxel multi-omics case study.

The filters in this module are intentionally column-selection helpers rather
than fitted models. In cross-validation, call them on the training fold only,
then apply the returned columns to both train and test matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


CONTINUOUS_RESPONSE_COLS = ["angle", "TGI"]
CLASSIFICATION_RESPONSE_COLS = ["mRECIST", "response_binary", "response_binary_num"]
RESPONSE_COLS = CONTINUOUS_RESPONSE_COLS + CLASSIFICATION_RESPONSE_COLS
BINARY_RESPONSE_MAP = {
    "PD": "Resistant",
    "SD": "Resistant",
    "PR": "Sensitive",
    "CR": "Sensitive",
}
BINARY_RESPONSE_NUM_MAP = {
    "Resistant": 0,
    "Sensitive": 1,
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAX_GENE_PATH = PROJECT_ROOT / "data/procdata/4_paclitaxel/pax_gene_list.tsv"
DEFAULT_NEST_GENE_PATH = PROJECT_ROOT / "data/rawdata/nest_vnn/gene2ind.txt"


def load_pax_genes(path: str | Path = DEFAULT_PAX_GENE_PATH) -> set[str]:
    """Load the curated paclitaxel-prior gene list."""
    genes = pd.read_csv(path, sep="\t")["Gene"].dropna().astype(str)
    return set(genes)


def load_nest_genes(path: str | Path = DEFAULT_NEST_GENE_PATH) -> set[str]:
    """Load the NEST-VNN gene prior from a gene2ind.txt file."""
    genes: set[str] = set()
    with Path(path).open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                genes.add(parts[1])
    return genes


def add_binary_response(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary paclitaxel response labels from mRECIST.

    Resistant = PD + SD; Sensitive = PR + CR.
    """
    if "mRECIST" not in df.columns:
        raise ValueError("Column 'mRECIST' is required to define binary response.")

    df = df.copy()
    df["response_binary"] = df["mRECIST"].map(BINARY_RESPONSE_MAP)
    df["response_binary_num"] = df["response_binary"].map(BINARY_RESPONSE_NUM_MAP)

    if df["response_binary_num"].isna().any():
        bad = df.loc[df["response_binary_num"].isna(), "mRECIST"].unique()
        raise ValueError(f"Unexpected mRECIST labels: {bad}")

    return df


def split_xy(df: pd.DataFrame, target: str = "TGI") -> tuple[pd.DataFrame, pd.Series]:
    """Split a merged omics-response table into X and a single response."""
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} is not present.")
    drop_cols = [col for col in RESPONSE_COLS if col in df.columns]
    return df.drop(columns=drop_cols), df[target]


def split_x_class_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a merged omics-response table into X and binary response y."""
    if "response_binary_num" not in df.columns:
        df = add_binary_response(df)
    drop_cols = [col for col in RESPONSE_COLS if col in df.columns]
    return df.drop(columns=drop_cols), df["response_binary_num"].astype(int)


def get_prior_genes(
    mode: str,
    pax_genes: Iterable[str],
    nest_genes: Iterable[str],
) -> set[str] | None:
    """Return the external prior gene set requested by mode."""
    if mode == "none":
        return None
    if mode == "pax":
        return set(pax_genes)
    if mode == "nest":
        return set(nest_genes)
    if mode == "pax_nest":
        return set(pax_genes) | set(nest_genes)
    raise ValueError(f"Unknown prior mode: {mode}")


def variance_filter(
    X: pd.DataFrame,
    n_top: int | None = None,
    var_threshold: float | None = None,
) -> list[str]:
    """Return columns selected by variance, using only the provided matrix."""
    variances = X.var(axis=0, numeric_only=True)

    if var_threshold is not None:
        selected = variances[variances > var_threshold].index
    elif n_top is not None:
        selected = variances.nlargest(min(n_top, len(variances))).index
    else:
        selected = variances[variances > 0].index

    return list(selected)


def corr_filter(X: pd.DataFrame, threshold: float = 0.85) -> list[str]:
    """Return columns after dropping later columns from highly correlated pairs."""
    if X.shape[1] <= 1:
        return list(X.columns)

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = {column for column in upper.columns if (upper[column] > threshold).any()}
    return [column for column in X.columns if column not in to_drop]


def mutation_prevalence_filter(
    X: pd.DataFrame,
    min_count: int = 10,
    max_fraction: float = 0.90,
) -> list[str]:
    """Return binary mutation columns within recurrence bounds."""
    mutation_count = X.sum(axis=0)
    max_count = int(max_fraction * X.shape[0])
    selected = mutation_count[
        (mutation_count >= min_count) & (mutation_count <= max_count)
    ].index
    return list(selected)


def force_retain_prior_genes(
    selected_genes: Iterable[str],
    prior_genes: Iterable[str],
    available_genes: Sequence[str],
    max_features: int | None = None,
) -> list[str]:
    """Add available prior genes to a selected feature set.

    If max_features is provided, all available prior genes are kept first and
    non-prior selected genes are added in their original selected order until
    the feature budget is reached.
    """
    selected = list(dict.fromkeys(selected_genes))
    prior = set(prior_genes)
    available = list(available_genes)
    available_set = set(available)

    prior_available = [gene for gene in available if gene in prior]

    if max_features is None:
        final = list(dict.fromkeys(selected + prior_available))
        return [gene for gene in final if gene in available_set]

    if max_features < len(prior_available):
        raise ValueError(
            f"max_features={max_features} is smaller than the number of "
            f"available prior genes ({len(prior_available)})."
        )

    non_prior_selected = [gene for gene in selected if gene not in prior and gene in available_set]
    final = prior_available + non_prior_selected[: max_features - len(prior_available)]
    return list(dict.fromkeys(final))


def select_rna_features(
    X_train: pd.DataFrame,
    prior_genes: Iterable[str] | None = None,
    n_prefilter: int = 2200,
    corr_threshold: float = 0.85,
    n_top: int = 2000,
    max_features: int | None = None,
) -> list[str]:
    """Fold-safe RNA feature selection for a training matrix."""
    selected = variance_filter(X_train, n_top=n_prefilter)
    selected = corr_filter(X_train[selected], threshold=corr_threshold)
    selected = variance_filter(X_train[selected], n_top=n_top)

    if prior_genes is not None:
        selected = force_retain_prior_genes(
            selected,
            prior_genes=prior_genes,
            available_genes=X_train.columns,
            max_features=max_features,
        )
    return selected


def select_cnv_features(
    X_train: pd.DataFrame,
    prior_genes: Iterable[str] | None = None,
    corr_threshold: float = 0.95,
    n_top: int | None = None,
    max_features: int | None = None,
) -> list[str]:
    """Fold-safe CNV feature selection for a training matrix."""
    selected = variance_filter(X_train)
    selected = corr_filter(X_train[selected], threshold=corr_threshold)
    if n_top is not None:
        selected = variance_filter(X_train[selected], n_top=n_top)

    if prior_genes is not None:
        selected = force_retain_prior_genes(
            selected,
            prior_genes=prior_genes,
            available_genes=X_train.columns,
            max_features=max_features,
        )
    return selected


def select_mutation_features(
    X_train: pd.DataFrame,
    prior_genes: Iterable[str] | None = None,
    min_count: int = 10,
    max_fraction: float = 0.90,
    max_features: int | None = None,
) -> list[str]:
    """Fold-safe mutation feature selection for a training matrix."""
    selected = mutation_prevalence_filter(
        X_train,
        min_count=min_count,
        max_fraction=max_fraction,
    )

    if prior_genes is not None:
        selected = force_retain_prior_genes(
            selected,
            prior_genes=prior_genes,
            available_genes=X_train.columns,
            max_features=max_features,
        )
    return selected


def select_features_by_omics(
    omics_name: str,
    X_train: pd.DataFrame,
    prior_mode: str = "pax",
    pax_genes: Iterable[str] | None = None,
    nest_genes: Iterable[str] | None = None,
) -> list[str]:
    """Select features for one omics layer using the case-study defaults."""
    pax_genes = set() if pax_genes is None else set(pax_genes)
    nest_genes = set() if nest_genes is None else set(nest_genes)
    prior = get_prior_genes(prior_mode, pax_genes=pax_genes, nest_genes=nest_genes)

    if omics_name.startswith("rna"):
        return select_rna_features(
            X_train,
            prior_genes=prior,
            n_prefilter=2200,
            corr_threshold=0.85,
            n_top=2000,
        )

    if omics_name == "cnv":
        return select_cnv_features(
            X_train,
            prior_genes=prior,
            corr_threshold=0.95,
            n_top=None,
        )

    if omics_name == "mutation":
        return select_mutation_features(
            X_train,
            prior_genes=prior,
            min_count=10,
            max_fraction=0.90,
        )

    raise ValueError(f"Unknown omics name: {omics_name}")
