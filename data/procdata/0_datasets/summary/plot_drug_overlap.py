"""Bar chart summarizing drug_dataset_overlap.csv: drugs unique to each dataset
vs. drugs shared across datasets.

Simpler alternative to a 4-set Venn/UpSet plot -- the actual overlap here is
sparse (only one dataset pair shares any drugs, no 3- or 4-way overlaps), so
a bar per "unique to X" / "shared between X and Y" group tells the same
story without extra machinery.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "drug_dataset_overlap.csv"

DATASET_ORDER = ["mcgill_breast", "uhn_breast", "uhn_lung", "pdxe"]
DATASET_LABELS = {
    "mcgill_breast": "McGill breast",
    "uhn_breast": "UHN breast",
    "uhn_lung": "UHN lung",
    "pdxe": "PDXE pan-cancer",
}
UNIQUE_COLOR = "#F6BEC0"
SHARED_COLOR = "#D9534F"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_overlap_counts(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    return df["datasets"].value_counts()


def group_label(datasets_key):
    parts = datasets_key.split(",")
    if len(parts) == 1:
        return f"{DATASET_LABELS[parts[0]]} only"
    return " + ".join(DATASET_LABELS[p] for p in parts)


def plot_drug_overlap(counts, out_name="barplot_drug_overlap", out_dir=HERE):
    # Single-dataset bars first (in DATASET_ORDER), shared combos last.
    single_keys = [d for d in DATASET_ORDER if d in counts.index]
    shared_keys = [k for k in counts.index if k not in DATASET_ORDER]
    ordered_keys = single_keys + shared_keys

    labels = [group_label(k) for k in ordered_keys]
    values = [int(counts[k]) for k in ordered_keys]
    colors = [UNIQUE_COLOR if k in DATASET_ORDER else SHARED_COLOR for k in ordered_keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = range(len(labels))
    ax.barh(y, values, height=0.55, color=colors, edgecolor="black", linewidth=0.6)
    ax.invert_yaxis()

    for yi, value in zip(y, values):
        ax.text(value + max(values) * 0.02, yi, f"{value}", va="center", fontsize=14)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Drugs")
    ax.set_xlim(0, max(values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)

    total = sum(values)
    ax.text(
        0.99, -0.16, f"Total n={total}", transform=ax.transAxes,
        ha="right", va="top", fontsize=13,
    )

    plt.tight_layout()
    fig.savefig(out_dir / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / (out_name + '.png')}")


def main():
    counts = load_overlap_counts()
    print(counts)
    plot_drug_overlap(counts)


if __name__ == "__main__":
    main()
