from pathlib import Path

import pandas as pd

# Explicit functional variant classifications treated as a mutation "hit"
# (same convention as 3_carboplatin/pipeline_utils.py, applied here for
# consistency across case studies).
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


def binarize_mutation_inclusive(val):
    """
    Binarizes mutation data by checking for the presence of
    explicitly functional variant classifications.
    """
    if pd.isna(val) or val == "" or str(val).lower() == "nan":
        return 0

    parts = {p.strip() for p in str(val).split(',')}

    if any(p in FUNCTIONAL_WHITELIST for p in parts):
        return 1

    return 0


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


def make_case4_paths(project_dir=None):
    project_dir = find_project_root() if project_dir is None else Path(project_dir)

    paths = {
        "project": project_dir,
        "raw": project_dir / "rawdata" / "uhn_breast",
        "proc": project_dir / "results" / "4_paclitaxel",
        "datasets": project_dir / "procdata",
        "results": project_dir / "results" / "4_paclitaxel",
        "figures": project_dir / "figures_tables" / "figures_main",
        "figures_supp": project_dir / "figures_tables" / "figures_supp",
        "tables": project_dir / "figures_tables" / "tables",
    }

    paths["proc"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    paths["figures_supp"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)

    return paths
