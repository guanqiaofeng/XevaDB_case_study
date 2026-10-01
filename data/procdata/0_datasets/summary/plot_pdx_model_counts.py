"""Stacked horizontal bar plot of parental vs. resistant PDX model counts by cohort.

Parental/resistant split is derived directly from each dataset's models.csv
(resistant sublines are the rows whose model identifier contains "RES"),
deduplicated to the same unique-model granularity as the "Model" row of
dataset_summary.xlsx -- each dataset encodes its unique model differently, so
each needs its own extraction rule:
  - McGill breast / UHN lung: one model per unique patient.id.
  - UHN breast: model.id is "<base>.<drug>.<mouse>" -- one model per base
    id before the first ".".
  - PDXE pan-cancer: model.id is "X.<line>.<condition>" -- one model per
    base tumor line (the middle segment).
None of McGill breast, UHN lung, or PDXE pan-cancer have any
resistant-labeled model; UHN breast is the only cohort with resistant
sublines.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATASETS_DIR = HERE.parent

COHORT_ORDER = ["McGill breast", "UHN breast", "UHN lung", "PDXE pan-cancer"]
PARENTAL_COLOR = "#6E893B"
RESISTANT_COLOR = "#C9D965"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def _split_count(base_ids):
    uniq_base = pd.Series(pd.Series(base_ids).unique())
    resistant = int(uniq_base.str.contains("RES", case=False, na=False).sum())
    parental = int(len(uniq_base) - resistant)
    return parental, resistant


def count_by_patient(models_csv):
    """McGill breast / UHN lung: one model per unique patient.id."""
    df = pd.read_csv(models_csv)
    return _split_count(df["patient.id"])


def count_by_first_dot(models_csv):
    """UHN breast: model.id is "<base>.<drug>.<mouse>" -- base is everything before
    the last two dot-separated segments (drug, mouse), not just before the first dot.
    One subline's own base name happens to contain a "." itself
    (BXTO.64_P7_ORG_P2_ERIBLD_210603_RES.H2O.m1 -- base is
    "BXTO.64_P7_ORG_P2_ERIBLD_210603_RES", a resistant line), which a naive
    first-dot split would truncate to "BXTO" and silently drop the "_RES"
    marker, miscounting it as parental.
    """
    df = pd.read_csv(models_csv)
    return _split_count(df["model.id"].str.rsplit(".", n=2).str[0])


def count_by_middle_segment(models_csv):
    """PDXE: model.id is "X.<line>.<condition>" -- base tumor line is the middle segment."""
    df = pd.read_csv(models_csv)
    parts = df["model.id"].str.split(".", n=2, expand=True)
    return _split_count(parts[1])


def load_model_counts():
    sources = {
        "McGill breast": (DATASETS_DIR / "mcgill_breast" / "models.csv", count_by_patient),
        "UHN breast": (DATASETS_DIR / "uhn_breast" / "models.csv", count_by_first_dot),
        "UHN lung": (DATASETS_DIR / "uhn_lung" / "models.csv", count_by_patient),
        "PDXE pan-cancer": (DATASETS_DIR / "pdxe" / "csv" / "models.csv", count_by_middle_segment),
    }
    rows = []
    for cohort, (path, count_fn) in sources.items():
        parental, resistant = count_fn(path)
        rows.append({"cohort": cohort, "parental": parental, "resistant": resistant})
    return pd.DataFrame(rows).set_index("cohort").loc[COHORT_ORDER]


def plot_model_counts(counts, out_name="barplot_pdx_models", out_dir=HERE):
    fig, ax = plt.subplots(figsize=(4, 3.5))

    parental = counts["parental"]
    resistant = counts["resistant"]
    total = parental + resistant

    ax.barh(
        COHORT_ORDER, parental, color=PARENTAL_COLOR, height=0.5, label="Parental",
        edgecolor="black", linewidth=0.6,
    )
    ax.barh(
        COHORT_ORDER, resistant, left=parental,
        color=RESISTANT_COLOR, height=0.5, label="Resistant",
        edgecolor="black", linewidth=0.6,
    )
    ax.invert_yaxis()  # first cohort (McGill breast) on top

    for cohort, p, r, t in zip(COHORT_ORDER, parental, resistant, total):
        # total, at the tip of the full stacked bar
        ax.text(t + total.max() * 0.02, cohort, f"{int(t)}", va="center", fontsize=16)
        # resistant sub-count, centered in the resistant segment (UHN breast only)
        if cohort == "UHN breast" and r > 0:
            ax.text(
                p + r / 2, cohort, f"{int(r)}",
                va="center", ha="center", fontsize=13, color="black",
            )

    ax.set_xlabel("PDX models")
    ax.set_xlim(0, total.max() * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        frameon=False, fontsize=14,
    )

    plt.tight_layout()
    fig.savefig(out_dir / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / (out_name + '.png')}")


def main():
    counts = load_model_counts()
    print(counts)
    plot_model_counts(counts)


if __name__ == "__main__":
    main()
