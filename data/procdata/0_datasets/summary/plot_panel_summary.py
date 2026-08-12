"""Combined panel: Patient / PDX / Omics / Drug / Experiment, side by side.

Reuses the data-loading helpers from the individual plot_*.py scripts in this
folder, but draws its own compact bars onto shared subplot axes so titles
replace the individual x-axis labels and only the leftmost panel keeps
cohort labels.
"""

import numpy as np
import matplotlib.pyplot as plt

from plot_dataset_summary import HERE, COHORT_ORDER, BAR_COLOR, load_summary
from plot_pdx_model_counts import load_model_counts, PARENTAL_COLOR, RESISTANT_COLOR
from plot_omics_counts import OMICS_TYPES, OMICS_COLORS
from plot_experiment_counts import METRIC_COLORS as EXPERIMENT_COLORS
from plot_drug_overlap import load_overlap_counts

DRUG_COLOR = "#F6BEC0"
GROUP_BAR_HEIGHT = 0.22
GROUP_BAR_SPACING = 0.30
SINGLE_BAR_HEIGHT = 0.5
XLIM_PAD_FACTOR = 1.3  # every draw_* function sets xlim to (0, max_value * this)


def draw_single_metric(ax, df, metric, color, cohort_order):
    values = df.loc[metric, cohort_order]
    y = np.arange(len(cohort_order))
    ax.barh(y, values, height=SINGLE_BAR_HEIGHT, color=color, edgecolor="black", linewidth=0.6)
    for yi, value in zip(y, values):
        ax.text(value + values.max() * 0.04, yi, f"{int(value)}", va="center", fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels(cohort_order)
    ax.invert_yaxis()
    ax.set_xlim(0, values.max() * XLIM_PAD_FACTOR)


def draw_pdx(ax, counts, cohort_order):
    y = np.arange(len(cohort_order))
    parental = counts["parental"]
    resistant = counts["resistant"]
    total = parental + resistant

    ax.barh(y, parental, height=SINGLE_BAR_HEIGHT, color=PARENTAL_COLOR,
            edgecolor="black", linewidth=0.6, label="Parental model")
    ax.barh(y, resistant, left=parental, height=SINGLE_BAR_HEIGHT, color=RESISTANT_COLOR,
            edgecolor="black", linewidth=0.6, label="Resistant model")

    for yi, cohort, p, r, t in zip(y, cohort_order, parental, resistant, total):
        ax.text(t + total.max() * 0.04, yi, f"{int(t)}", va="center", fontsize=14)
        if cohort == "UHN breast":
            if r > 0:
                ax.text(p + r / 2, yi, f"{int(r)}", va="center", ha="center", fontsize=13, color="black")
            if p > 0:
                ax.text(p / 2, yi, f"{int(p)}", va="center", ha="center", fontsize=13, color="white")

    ax.set_yticks(y)
    ax.set_yticklabels(cohort_order)
    ax.invert_yaxis()
    ax.set_xlim(0, total.max() * XLIM_PAD_FACTOR)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=1, frameon=False, fontsize=13)


def draw_grouped(ax, df, metrics, colors, cohort_order):
    y_base = np.arange(len(cohort_order))
    offsets = np.linspace(-1, 1, len(metrics)) * GROUP_BAR_SPACING
    max_value = df.loc[list(metrics), cohort_order].to_numpy().max()

    for (metric, label), offset in zip(metrics.items(), offsets):
        values = df.loc[metric, cohort_order]
        y = y_base + offset
        ax.barh(y, values, height=GROUP_BAR_HEIGHT, color=colors[metric],
                edgecolor="black", linewidth=0.6, label=label)
        for yi, value in zip(y, values):
            ax.text(value + max_value * 0.04, yi, f"{int(value)}", va="center", fontsize=12)

    ax.set_yticks(y_base)
    ax.set_yticklabels(cohort_order)
    ax.invert_yaxis()
    ax.set_xlim(0, max_value * XLIM_PAD_FACTOR)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=1, frameon=False, fontsize=13)


def style_panel(ax, title, show_ylabels):
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)
    if not show_ylabels:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)


def n_label(df, *metrics):
    """"n=" label from the xlsx's own Total column, comma-formatted, slash-joined
    for panels with more than one metric."""
    parts = [f"{int(df.loc[metric, 'Total']):,}" for metric in metrics]
    return "n=" + "/".join(parts)


def main():
    df = load_summary()
    pdx_counts = load_model_counts()
    omics_map = {m: m for m in OMICS_TYPES}

    fig, axes = plt.subplots(1, 5, figsize=(17, 5))

    draw_single_metric(axes[0], df, "Patient", BAR_COLOR, COHORT_ORDER)
    draw_pdx(axes[1], pdx_counts, COHORT_ORDER)
    draw_grouped(axes[2], df, omics_map, OMICS_COLORS, COHORT_ORDER)
    draw_single_metric(axes[3], df, "Drug", DRUG_COLOR, COHORT_ORDER)
    draw_single_metric(
        axes[4], df, "treatment-control experiment",
        EXPERIMENT_COLORS["treatment curves"], COHORT_ORDER,
    )

    titles = ["Patient", "PDX", "Omics", "Drug", "Experiment"]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        style_panel(ax, title, show_ylabels=(i == 0))

    plt.tight_layout()

    drug_total = int(load_overlap_counts().sum())

    n_labels = [
        n_label(df, "Patient"),
        n_label(df, "Model"),
        n_label(df, "RNAseq", "Mutation", "CNV"),
        f"n={drug_total:,}",
        n_label(df, "treatment-control experiment"),
    ]
    # All 5 axes share the same row/height, so a fixed axes-fraction y aligns
    # every "n=" label on the same line -- immediately below the bars and
    # above each panel's (now pushed further down) legend. Centered on the
    # actual bar range (data 0 to max_value), not the padded axes width
    # (0 to max_value * XLIM_PAD_FACTOR) -- otherwise the label sits right
    # of the bars' visual center, since the padding is one-sided (right only).
    N_LABEL_Y = -0.14
    N_LABEL_X = 1 / (2 * XLIM_PAD_FACTOR)
    for ax, label in zip(axes, n_labels):
        ax.text(N_LABEL_X, N_LABEL_Y, label, transform=ax.transAxes, ha="center", va="top", fontsize=17)

    # Single "Total" row header, in the same left column as the dataset
    # names and on the same line as the "n=" values.
    axes[0].text(
        -0.35, N_LABEL_Y, "Total", transform=axes[0].transAxes,
        ha="right", va="top", fontsize=17,
    )

    out_name = "panel_dataset_summary"
    fig.savefig(HERE / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(HERE / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {HERE / (out_name + '.png')}")


if __name__ == "__main__":
    main()
