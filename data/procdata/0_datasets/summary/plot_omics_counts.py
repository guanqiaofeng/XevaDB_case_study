"""Grouped horizontal bar plot of RNAseq/Mutation/CNV sample counts by cohort."""

import numpy as np
import matplotlib.pyplot as plt

from plot_dataset_summary import HERE, COHORT_ORDER, load_summary

OMICS_TYPES = ["RNAseq", "Mutation", "CNV"]
OMICS_COLORS = {
    "RNAseq": "#8A6915",
    "Mutation": "#D9A421",
    "CNV": "#FBBF29",
}
BAR_HEIGHT = 0.24
BAR_SPACING = 0.32


def plot_omics_barplot(df, out_name="barplot_omics", out_dir=HERE, cohort_order=COHORT_ORDER):
    y_base = np.arange(len(cohort_order))
    offsets = np.linspace(-1, 1, len(OMICS_TYPES)) * BAR_SPACING
    max_value = df.loc[OMICS_TYPES, cohort_order].to_numpy().max()

    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    for omics, offset in zip(OMICS_TYPES, offsets):
        values = df.loc[omics, cohort_order]
        y = y_base + offset
        ax.barh(
            y, values, height=BAR_HEIGHT, color=OMICS_COLORS[omics],
            edgecolor="black", linewidth=0.6, label=omics,
        )
        for yi, value in zip(y, values):
            ax.text(value + max_value * 0.02, yi, f"{int(value)}", va="center", fontsize=12)

    ax.set_yticks(y_base)
    ax.set_yticklabels(cohort_order)
    ax.invert_yaxis()  # first cohort (McGill breast) on top

    ax.set_xlabel("Samples")
    ax.set_xlim(0, max_value * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
        frameon=False, fontsize=13,
    )

    plt.tight_layout()
    fig.savefig(out_dir / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / (out_name + '.png')}")


def main():
    df = load_summary()
    plot_omics_barplot(df)


if __name__ == "__main__":
    main()
