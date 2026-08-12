"""Grouped horizontal bar plot of treatment-control experiment/treatment/control
curve counts by cohort.
"""

import numpy as np
import matplotlib.pyplot as plt

from plot_dataset_summary import HERE, COHORT_ORDER, load_summary

# xlsx row name -> display label
METRICS = {
    "treatment-control experiment": "Treatment-control experiment",
    "treatment curves": "Treatment curves",
    "control curves": "Control curves",
}
METRIC_COLORS = {
    "treatment-control experiment": "#8A844D",
    "treatment curves": "#CCC674",
    "control curves": "#EBE482",
}
BAR_HEIGHT = 0.24
BAR_SPACING = 0.32


def plot_experiment_barplot(df, out_name="barplot_experiments", out_dir=HERE, cohort_order=COHORT_ORDER):
    y_base = np.arange(len(cohort_order))
    offsets = np.linspace(-1, 1, len(METRICS)) * BAR_SPACING
    max_value = df.loc[list(METRICS), cohort_order].to_numpy().max()

    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    for (metric, label), offset in zip(METRICS.items(), offsets):
        values = df.loc[metric, cohort_order]
        y = y_base + offset
        ax.barh(
            y, values, height=BAR_HEIGHT, color=METRIC_COLORS[metric],
            edgecolor="black", linewidth=0.6, label=label,
        )
        for yi, value in zip(y, values):
            ax.text(value + max_value * 0.02, yi, f"{int(value)}", va="center", fontsize=12)

    ax.set_yticks(y_base)
    ax.set_yticklabels(cohort_order)
    ax.invert_yaxis()  # first cohort (McGill breast) on top

    ax.set_xlabel("Count")
    ax.set_xlim(0, max_value * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
        frameon=False, fontsize=13,
    )

    plt.tight_layout()
    fig.savefig(out_dir / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / (out_name + '.png')}")


def main():
    df = load_summary()
    plot_experiment_barplot(df)


if __name__ == "__main__":
    main()
