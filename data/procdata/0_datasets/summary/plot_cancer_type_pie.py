"""Pie chart of PDX model counts by cancer type / dataset (Sheet2 of dataset_summary.xlsx)."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Wedge

from plot_dataset_summary import HERE, XLSX_PATH

EXPLODE_DISTANCE = 0.09

sns.set_theme(style="white", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Ordered so same-cancer-type slices sit adjacent in the pie. Colors echo
# ones already used across the summary panel for visual consistency:
# McGill breast reuses the Drug pink (#F6BEC0), PDXE lung reuses the
# Patient blue (#64A8DC), colorectal reuses the PDX Parental green
# (#6E893B), and melanoma reuses the Omics Mutation gold (#D9A421).
DATASET_ORDER = [
    "UHN breast", "PDXE breast", "McGill breast",
    "UHN lung", "PDXE lung",
    "PDXE gastric",
    "PDXE colorectal",
    "PDXE pancreatic",
    "PDXE cutaneous melanoma",
]
DATASET_COLORS = {
    "UHN breast": "#D44780",
    "PDXE breast": "#E8729A",
    "McGill breast": "#F6BEC0",
    "UHN lung": "#2E5C8A",
    "PDXE lung": "#64A8DC",
    "PDXE gastric": "#F4D03F",
    "PDXE colorectal": "#9AC339",
    "PDXE pancreatic": "#9F7BC2",
    "PDXE cutaneous melanoma": "#FAB811",
}


def load_cancer_type_counts(xlsx_path=XLSX_PATH):
    df = pd.read_excel(xlsx_path, sheet_name="Sheet2")
    df = df[df["Dataset"] != "Total"]
    return df.set_index("Dataset").loc[DATASET_ORDER]


def plot_cancer_type_pie(df, out_name="pie_cancer_type", out_dir=HERE):
    # Cancer types in first-seen order (DATASET_ORDER already groups same-type
    # datasets together). Every wedge keeps its true proportional angle (the
    # full set still sweeps a complete 360 degrees, no angle is discarded) --
    # each cancer type's whole cluster of wedges is just translated outward,
    # together, along that group's own mid-angle direction, so groups pull
    # apart from a shared center like an exploded pie slice.
    cancer_types = list(dict.fromkeys(df["CancerType"]))
    grand_total = df["Model"].sum()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.35, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")

    wedge_patches = []
    wedge_labels = []
    angle = 90.0  # start at 12 o'clock, sweep counterclockwise, full 360 total

    for group_i, cancer_type in enumerate(cancer_types):
        group_df = df[df["CancerType"] == cancer_type]
        group_span = 360 * (group_df["Model"].sum() / grand_total)
        group_start = angle
        group_end = angle + group_span
        group_mid = np.deg2rad((group_start + group_end) / 2)
        center = (EXPLODE_DISTANCE * np.cos(group_mid), EXPLODE_DISTANCE * np.sin(group_mid))

        sub_angle = group_start
        for dataset, row in group_df.iterrows():
            share = group_span * (row["Model"] / group_df["Model"].sum())
            theta1, theta2 = sub_angle, sub_angle + share

            wedge = Wedge(
                center, 1.0, theta1, theta2,
                facecolor=DATASET_COLORS[dataset], edgecolor="black", linewidth=0.8,
            )
            ax.add_patch(wedge)
            wedge_patches.append(wedge)
            wedge_labels.append(f"{dataset} (n={int(row['Model'])})")

            sub_angle = theta2

        # Callout line + label at the group's own mid-angle, naming the
        # cancer type and its total model count across all its datasets.
        # Radius is staggered (near/far) between consecutive groups so
        # labels on closely-spaced small slices don't collide.
        line_radius = 1.08 if group_i % 2 == 0 else 1.22
        line_start = (center[0] + 0.98 * np.cos(group_mid), center[1] + 0.98 * np.sin(group_mid))
        line_end = (center[0] + line_radius * np.cos(group_mid), center[1] + line_radius * np.sin(group_mid))
        ax.plot(
            [line_start[0], line_end[0]], [line_start[1], line_end[1]],
            color="black", linewidth=1,
        )
        group_total = int(group_df["Model"].sum())
        short_type = cancer_type.replace(" cancer", "")
        text_ha = "left" if np.cos(group_mid) >= 0 else "right"
        ax.text(
            line_end[0] + 0.02 * np.cos(group_mid), line_end[1] + 0.02 * np.sin(group_mid),
            f"{short_type} ({group_total})",
            ha=text_ha, va="center", fontsize=16,
        )

        angle = group_end

    ax.legend(
        wedge_patches, wedge_labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, frameon=False, fontsize=14,
    )

    plt.tight_layout()
    fig.savefig(out_dir / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / (out_name + '.png')}")


def main():
    df = load_cancer_type_counts()
    plot_cancer_type_pie(df)


if __name__ == "__main__":
    main()
