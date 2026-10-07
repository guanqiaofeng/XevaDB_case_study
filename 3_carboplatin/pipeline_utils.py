from pathlib import Path

import pandas as pd

# Explicit functional variant classifications treated as a mutation "hit"
FUNCTIONAL_WHITELIST = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Translation_Start_Site",
    "Nonstop_Mutation",
}


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


def make_case3_paths(project_dir=None):
    project_dir = find_project_root() if project_dir is None else Path(project_dir)

    paths = {
        "project": project_dir,
        "raw": project_dir / "rawdata",
        "datasets": project_dir / "procdata",
        "proc": project_dir / "results" / "3_carboplatin",
        "results": project_dir / "results" / "3_carboplatin",
        "figures": project_dir / "figures_tables" / "figures_main",
        "figures_supp": project_dir / "figures_tables" / "figures_supp",
        "tables": project_dir / "figures_tables" / "tables",
    }

    paths["proc"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["figures_supp"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)

    return paths


# Batch-level mRECIST via majority vote across a batch's treatment-arm mice,
# ties broken toward the more resistant call. Shared by notebooks 1, 2, and 3.
MRECIST_WORST_FIRST = ["PD", "SD", "PR", "CR"]


def majority_vote_mrecist(values):
    counts = values.dropna().value_counts()
    if counts.empty:
        return None
    top_count = counts.max()
    tied = counts[counts == top_count].index
    return next(r for r in MRECIST_WORST_FIRST if r in tied)


def binarize_mutation_inclusive(val):
    """
    Binarizes mutation data by checking for the presence of
    explicitly functional variant classifications.
    """
    if pd.isna(val) or val == "" or str(val).lower() == "nan":
        return 0

    # Split by comma for cases where multiple mutations are collapsed into one cell
    # Strip whitespace and ensure we handle potential case sensitivity issues
    parts = {p.strip() for p in str(val).split(',')}

    # INCLUSIVE LOGIC: If ANY of the parts are in the whitelist, it's a 1
    if any(p in FUNCTIONAL_WHITELIST for p in parts):
        return 1

    # Otherwise (e.g., only "Silent", "Intron", or unknown terms), return 0
    return 0


def matrix_stats(name, df):
    total = df.size
    nonzero = (df != 0).sum().sum()
    sparsity = 1 - nonzero / total

    print(f"\n{name} statistics")
    print("----------------------------")
    print("Models:", df.shape[0])
    print("Genes:", df.shape[1])
    print("Total entries:", total)
    print("Nonzero entries:", nonzero)
    print("Sparsity:", round(sparsity, 4))

    model_events = (df != 0).sum(axis=1)
    gene_events = (df != 0).sum(axis=0)

    print("\nPer model events")
    print("Median:", model_events.median())
    print("Mean:", round(model_events.mean(), 2))
    print("Max:", model_events.max())

    print("\nPer gene events")
    print("Median:", gene_events.median())
    print("Mean:", round(gene_events.mean(), 2))
    print("Max:", gene_events.max())
