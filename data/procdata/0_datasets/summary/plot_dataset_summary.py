"""Horizontal bar plots summarizing dataset_summary.xlsx counts by cohort."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
XLSX_PATH = HERE / "dataset_summary.xlsx"

COHORT_ORDER = ["McGill breast", "UHN breast", "UHN lung", "PDXE pan-cancer"]
BAR_COLOR = "#64A8DC"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_summary(xlsx_path=XLSX_PATH):
    return pd.read_excel(xlsx_path, sheet_name="Sheet1", index_col=0)


def plot_metric_barplot(
    df, metric, xlabel, out_name, cohort_order=COHORT_ORDER, out_dir=HERE, color=BAR_COLOR,
):
    values = df.loc[metric, cohort_order]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.barh(
        cohort_order, values, color=color, height=0.5,
        edgecolor="black", linewidth=0.6,
    )
    ax.invert_yaxis()  # first cohort (McGill breast) on top

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + values.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center", fontsize=16
        )

    ax.set_xlabel(xlabel)
    ax.set_xlim(0, values.max() * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)

    plt.tight_layout()
    fig.savefig(out_dir / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / (out_name + '.png')}")


def main():
    df = load_summary()
    plot_metric_barplot(df, "Patient", xlabel="Patient", out_name="barplot_patients")


if __name__ == "__main__":
    main()
