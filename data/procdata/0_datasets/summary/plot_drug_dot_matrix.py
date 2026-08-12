"""UpSet-style dot-matrix of the drugs shared across more than one dataset
(drug_dataset_overlap.csv). Only the shared drugs are shown as rows -- with
just 10 of them, listing each by name is clearer than collapsing into
per-combination bars like a full UpSet plot would.
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
DOT_COLOR = "black"
COLUMN_SHADE_COLORS = {
    "mcgill_breast": "#FCE0BE",
    "uhn_breast": "#C9DDF2",
    "uhn_lung": "#E3BDB6",
    "pdxe": "#D3E8C6",
}

sns.set_theme(style="white", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_shared_drugs(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df["dataset_list"] = df["datasets"].str.split(",")
    df["n_datasets"] = df["dataset_list"].apply(len)
    shared = df[df["n_datasets"] > 1].copy()
    shared = shared.sort_values(
        ["n_datasets", "drug"], ascending=[False, True], key=lambda s: s.str.lower() if s.name == "drug" else s,
    ).reset_index(drop=True)

    # Keep Cisplatin next to Carboplatin (both platinum agents) rather than
    # wherever plain alphabetical order would put it.
    drugs = shared["drug"].tolist()
    if "Cisplatin" in drugs and "Carboplatin" in drugs:
        drugs.remove("Cisplatin")
        drugs.insert(drugs.index("Carboplatin") + 1, "Cisplatin")
        shared = shared.set_index("drug").loc[drugs].reset_index()

    return shared


def plot_drug_dot_matrix(shared, out_name="dotmatrix_shared_drugs", out_dir=HERE):
    n_rows = len(shared)
    n_cols = len(DATASET_ORDER)

    fig, ax = plt.subplots(figsize=(6, (0.55 * n_rows + 1.5) * 0.9))

    for col_i, ds in enumerate(DATASET_ORDER):
        ax.axvspan(col_i - 0.5, col_i + 0.5, color=COLUMN_SHADE_COLORS[ds], linewidth=0, zorder=0)

    for row_i, row in shared.iterrows():
        present = set(row["dataset_list"])
        member_x = [col_i for col_i, ds in enumerate(DATASET_ORDER) if ds in present]

        # connecting line + solid dots across the columns this drug is in
        if len(member_x) > 1:
            ax.plot([min(member_x), max(member_x)], [row_i, row_i], color=DOT_COLOR, linewidth=2, zorder=2)
        ax.scatter(member_x, [row_i] * len(member_x), s=220, color=DOT_COLOR, zorder=3)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], rotation=45, ha="left")
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([d[:1].upper() + d[1:] for d in shared["drug"]])
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_dir / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / (out_name + '.png')}")


def main():
    shared = load_shared_drugs()
    print(shared[["drug", "n_datasets", "datasets"]])
    plot_drug_dot_matrix(shared)


if __name__ == "__main__":
    main()
