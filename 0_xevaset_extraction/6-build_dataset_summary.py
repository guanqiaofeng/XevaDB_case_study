"""
Generate dataset_summary.xlsx (Sheet1 + Sheet2) and drug_dataset_overlap.csv
from the 0_xevaset_extraction_summary CSV outputs, data summary needed for Figure 1b/1c/1d
"""

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATASETS_DIR = HERE.parent / "data" / "procdata" / "0_datasets"
SUMMARY_DIR = DATASETS_DIR / "summary"
XLSX_PATH = SUMMARY_DIR / "dataset_summary.xlsx"
OVERLAP_CSV_PATH = SUMMARY_DIR / "drug_dataset_overlap.csv"

VEHICLE_SUFFIX_PATTERNS = {
    "uhn_breast": re.compile(r"-(LOW|HD|\d+DAILY|\dON\dOFF)$"),
    "uhn_lung": re.compile(r"-\d+$"),
    "pdxe": re.compile(r"-[\d.]+mpk$", re.IGNORECASE),
}


def drug_set(names, exclude, suffix_pattern=None):
    """Individual compounds: combos split on '+', vehicles excluded, dose/schedule suffix stripped."""
    singles = set()
    for name in names:
        if name in exclude:
            continue
        for part in str(name).split("+"):
            part = part.strip()
            if suffix_pattern is not None:
                part = suffix_pattern.sub("", part)
            singles.add(part)
    return singles


def mcgill_metrics():
    path = DATASETS_DIR / "mcgill_breast"
    models = pd.read_csv(path / "models.csv")
    drug = pd.read_csv(path / "drug.csv")
    exp = pd.read_csv(path / "experiment.csv")
    ed = pd.read_csv(path / "expDesign.csv")
    mp = pd.read_csv(path / "modToBiobaseMap.csv")

    patient = models["patient.id"].nunique()
    mp_pdx = mp[mp["source"] == "PDX"]
    drugs = drug_set(drug["standard.name"], exclude={"untreated"})

    metrics = {
        "Patient": patient,
        "Model": patient,
        "Parental Model": patient,
        "Resistant Model": 0,
        "RNAseq": mp_pdx.loc[mp_pdx["mDataType"] == "RNAseq", "biobase.id"].nunique(),
        "Mutation": mp_pdx.loc[mp_pdx["mDataType"] == "mutation", "biobase.id"].nunique(),
        "CNV": mp_pdx.loc[mp_pdx["mDataType"] == "CNV", "biobase.id"].nunique(),
        "Drug": len(drugs),
        "Treatment": int((drug["standard.name"] != "untreated").sum()),
        "model-level treatment records": len(models),
        "treatment curves": exp.loc[exp["drug.id"] != "untreated", "model.id"].nunique(),
        "control curves": exp.loc[exp["drug.id"] == "untreated", "model.id"].nunique(),
        "treatment-control experiment": ed["batch.name"].nunique(),
    }
    cancer_types = [("McGill breast", "Breast Cancer", patient)]
    return metrics, drugs, cancer_types


def uhn_breast_metrics():
    path = DATASETS_DIR / "uhn_breast"
    models = pd.read_csv(path / "models.csv")
    drug = pd.read_csv(path / "drug.csv")
    exp = pd.read_csv(path / "experiment.csv")
    ed = pd.read_csv(path / "expDesign.csv")
    mp = pd.read_csv(path / "modToBiobaseMap.csv")

    lineage = models["model.id"].str.rsplit(".", n=2).str[0]
    mp_lineage = mp["model.id"].str.rsplit(".", n=2).str[0]
    unique_lineages = pd.Series(lineage.unique())
    is_resistant = unique_lineages.str.upper().str.contains("RES")

    patient = models["patient.id"].nunique()
    suffix = VEHICLE_SUFFIX_PATTERNS["uhn_breast"]
    drugs = drug_set(drug["drugname.standardized"], exclude={"H2O", "DMSO", "PEG400"}, suffix_pattern=suffix)

    metrics = {
        "Patient": patient,
        "Model": len(unique_lineages),
        "Parental Model": int((~is_resistant).sum()),
        "Resistant Model": int(is_resistant.sum()),
        "RNAseq": mp_lineage[mp["mDataType"] == "RNASeq"].nunique(),
        "Mutation": mp_lineage[mp["mDataType"] == "mutation"].nunique(),
        "CNV": mp_lineage[mp["mDataType"] == "CNV"].nunique(),
        "Drug": len(drugs),
        "Treatment": len(drug),
        "model-level treatment records": len(models),
        "treatment curves": exp.loc[exp["drug.id"] != "H2O", "model.id"].nunique(),
        "control curves": exp.loc[exp["drug.id"] == "H2O", "model.id"].nunique(),
        "treatment-control experiment": ed["batch.name"].nunique(),
    }
    cancer_types = [("UHN breast", "Breast Cancer", len(unique_lineages))]
    return metrics, drugs, cancer_types


def uhn_lung_metrics():
    path = DATASETS_DIR / "uhn_lung"
    models = pd.read_csv(path / "models.csv")
    drug = pd.read_csv(path / "drug.csv")
    exp = pd.read_csv(path / "experiment.csv")
    ed = pd.read_csv(path / "expDesign.csv")
    mp = pd.read_csv(path / "modToBiobaseMap.csv")

    patient = models["patient.id"].nunique()
    mp = mp.merge(models[["model.id", "patient.id"]].drop_duplicates(), on="model.id", how="left")
    suffix = VEHICLE_SUFFIX_PATTERNS["uhn_lung"]
    drugs = drug_set(drug["drugname.standardized"], exclude={"control"}, suffix_pattern=suffix)

    metrics = {
        "Patient": patient,
        "Model": patient,
        "Parental Model": patient,
        "Resistant Model": 0,
        "RNAseq": mp.loc[mp["mDataType"] == "RNASeq", "patient.id"].nunique(),
        "Mutation": mp.loc[mp["mDataType"] == "mutation", "patient.id"].nunique(),
        "CNV": mp.loc[mp["mDataType"] == "CNV", "patient.id"].nunique(),
        "Drug": len(drugs),
        "Treatment": int((drug["drugname.standardized"] != "untreated").sum()),
        "model-level treatment records": len(models),
        "treatment curves": exp.loc[exp["drug.id"] != "control", "model.id"].nunique(),
        "control curves": exp.loc[exp["drug.id"] == "control", "model.id"].nunique(),
        "treatment-control experiment": ed["batch.name"].nunique(),
    }
    cancer_types = [("UHN lung", "Non-small Cell Lung Carcinoma", patient)]
    return metrics, drugs, cancer_types


def pdxe_metrics():
    path = DATASETS_DIR / "pdxe" / "csv"
    models = pd.read_csv(path / "models.csv")
    drug = pd.read_csv(path / "drug.csv")
    ed = pd.read_csv(path / "expDesign.csv")
    mp = pd.read_csv(path / "modToBiobaseMap.csv")

    patient = models["patient.id"].nunique()
    suffix = VEHICLE_SUFFIX_PATTERNS["pdxe"]
    drugs = drug_set(drug["standard.name"], exclude={"untreated"}, suffix_pattern=suffix)

    pivot = ed.pivot_table(index="batch.name", columns="arm", values="model.id", aggfunc="count", fill_value=0)
    valid_pairs = int(((pivot.get("control", 0) > 0) & (pivot.get("treatment", 0) > 0)).sum())

    metrics = {
        "Patient": patient,
        "Model": patient,
        "Parental Model": patient,
        "Resistant Model": 0,
        "RNAseq": mp.loc[mp["mDataType"] == "RNASeq", "biobase.id"].nunique(),
        "Mutation": mp.loc[mp["mDataType"] == "mutation", "biobase.id"].nunique(),
        "CNV": mp.loc[mp["mDataType"] == "cnv", "biobase.id"].nunique(),
        "Drug": len(drugs),
        "Treatment": int((drug["standard.name"] != "untreated").sum()),
        "model-level treatment records": len(models),
        "treatment curves": ed.loc[ed["arm"] == "treatment", "model.id"].nunique(),
        "control curves": ed.loc[ed["arm"] == "control", "model.id"].nunique(),
        "treatment-control experiment": valid_pairs,
    }
    pdxe_labels = {
        "Breast Cancer": "breast",
        "Non-small Cell Lung Carcinoma": "lung",
        "Gastric Cancer": "gastric",
        "Colorectal Cancer": "colorectal",
        "Pancreatic Ductal Carcinoma": "pancreatic",
        "Cutaneous Melanoma": "cutaneous melanoma",
    }
    tissue_counts = models.groupby("tissue.name")["patient.id"].nunique()
    cancer_types = [(f"PDXE {pdxe_labels[t]}", t, n) for t, n in tissue_counts.items()]
    return metrics, drugs, cancer_types


ROW_ORDER = [
    "Patient", "Model", "Parental Model", "Resistant Model",
    "RNAseq", "Mutation", "CNV", "Drug", "Treatment",
    "model-level treatment records", "treatment curves", "control curves",
    "treatment-control experiment",
]
COHORT_FNS = {
    "McGill breast": ("mcgill_breast", mcgill_metrics),
    "UHN breast": ("uhn_breast", uhn_breast_metrics),
    "UHN lung": ("uhn_lung", uhn_lung_metrics),
    "PDXE pan-cancer": ("pdxe", pdxe_metrics),
}


def preferred_casing(variants):
    """Cohorts don't always agree on drug-name casing (UHN breast is
    ALL-CAPS, UHN lung/PDXE are lowercase INN-style). Where they disagree,
    prefer a mixed-case variant if any cohort has one; otherwise title-case
    a lowercase variant. Where they agree, keep that casing untouched.
    """
    if len(variants) == 1:
        return next(iter(variants))
    for v in variants:
        if not v.isupper() and not v.islower():
            return v
    for v in variants:
        if v.islower():
            return v.title()
    return next(iter(variants))


def build_sheet1_and_overlap():
    sheet1 = {}
    drug_membership = {}  # lowercased drug name -> {variants, cohorts}
    cancer_type_rows = []

    for cohort, (cohort_key, fn) in COHORT_FNS.items():
        metrics, drugs, cancer_types = fn()
        sheet1[cohort] = metrics
        cancer_type_rows.extend(cancer_types)
        for name in drugs:
            entry = drug_membership.setdefault(name.lower(), {"variants": set(), "cohorts": set()})
            entry["variants"].add(name)
            entry["cohorts"].add(cohort_key)

    df1 = pd.DataFrame(sheet1).loc[ROW_ORDER]
    df1.index.name = None
    df1["Total"] = df1.sum(axis=1)
    df1.loc["Drug", "Total"] = len(drug_membership)

    overlap_rows = sorted(
        (
            (preferred_casing(v["variants"]), ",".join(sorted(v["cohorts"])))
            for v in drug_membership.values()
        ),
        key=lambda r: r[0].lower(),
    )
    overlap_df = pd.DataFrame(overlap_rows, columns=["drug", "datasets"])

    sheet2 = pd.DataFrame(cancer_type_rows, columns=["Dataset", "CancerType", "Model"]).set_index("Dataset")
    sheet2.loc["Total"] = [None, sheet2["Model"].sum()]

    return df1, overlap_df, sheet2


def main():
    df1, overlap_df, sheet2 = build_sheet1_and_overlap()

    with pd.ExcelWriter(XLSX_PATH) as writer:
        df1.to_excel(writer, sheet_name="Sheet1")
        sheet2.to_excel(writer, sheet_name="Sheet2")

    overlap_df.to_csv(OVERLAP_CSV_PATH, index=False)
    print(f"wrote {XLSX_PATH}")
    print(f"wrote {OVERLAP_CSV_PATH}")


if __name__ == "__main__":
    main()
