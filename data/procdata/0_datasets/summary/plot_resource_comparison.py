"""Membership-style comparison matrix of PDX resources vs. capabilities
(Sheet3 of dataset_summary.xlsx). The "Primary focus" column is free-text,
not a capability, and is excluded. Cells are Yes/Limited/No -> solid dot /
semisolid dot / no dot. XevaDB's row label is bold.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_dataset_summary import HERE, XLSX_PATH

SHEET_NAME = "Sheet3"
RESOURCE_COL = "PDX Resource"
EXCLUDE_COLS = ["Primary focus", "Drug annotations"]
BOLD_RESOURCE = "XevaDB"

DOT_COLOR = "black"
DOT_MARKER = "o"
PARTIAL_ALPHA = 0.35
DOT_SIZE = 260
LABEL_WRAP_WIDTH = 11
# Explicit line breaks for labels where generic wrapping doesn't give the
# desired split.
LABEL_OVERRIDES = {
    "Longitudinal tumor volume": "Longitudinal\ntumor volume",
}
COLUMN_SHADE_COLORS = {
    "Model metadata": "#D6EAC4",
    "Molecular profiles": "#FCEEA6",
    "Drug response": "#CDE7F4",
    "Longitudinal tumor volume": "#7FB8D9",
    "ML-ready dataset": "#F5C6C6",
}

sns.set_theme(style="white", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_resource_matrix(xlsx_path=XLSX_PATH, sheet_name=SHEET_NAME):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = df.set_index(RESOURCE_COL)
    df = df.drop(columns=EXCLUDE_COLS)
    return df


def plot_resource_matrix(df, out_name="membership_pdx_resources", out_dir=HERE):
    resources = df.index.tolist()
    capabilities = df.columns.tolist()
    n_rows = len(resources)
    n_cols = len(capabilities)

    fig, ax = plt.subplots(figsize=(1.6 * n_cols + 2, 0.45 * n_rows + 2.2))

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))

    for col_i, capability in enumerate(capabilities):
        shade = COLUMN_SHADE_COLORS.get(capability)
        if shade is not None:
            ax.axvspan(col_i - 0.5, col_i + 0.5, color=shade, linewidth=0, zorder=0)

    for row_i, resource in enumerate(resources):
        for col_i, capability in enumerate(capabilities):
            value = str(df.loc[resource, capability]).strip().lower()
            if value == "yes":
                ax.scatter(col_i, row_i, s=DOT_SIZE, marker=DOT_MARKER, color=DOT_COLOR, zorder=2)
            elif value == "limited":
                ax.scatter(
                    col_i, row_i, s=DOT_SIZE, marker=DOT_MARKER, color=DOT_COLOR,
                    alpha=PARTIAL_ALPHA, zorder=2,
                )
            # "no" -> nothing drawn

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    wrapped_capabilities = [
        LABEL_OVERRIDES.get(c) or "\n".join(textwrap.wrap(c, width=LABEL_WRAP_WIDTH, break_long_words=False))
        for c in capabilities
    ]
    ax.set_xticklabels(wrapped_capabilities, rotation=0, ha="center")
    ax.xaxis.tick_top()
    ax.set_yticklabels(resources)

    for label, resource in zip(ax.get_yticklabels(), resources):
        if resource == BOLD_RESOURCE:
            label.set_fontweight("bold")

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / (out_name + '.png')}")


def main():
    df = load_resource_matrix()
    print(df)
    plot_resource_matrix(df)


if __name__ == "__main__":
    main()
