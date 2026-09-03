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
EXCLUDE_COLS = ["Portal website", "Organization"]
BOLD_RESOURCE = "XevaDB"

DOT_COLOR = "black"
DOT_MARKER = "o"
PARTIAL_ALPHA = 0.35
DOT_SIZE = 260
LABEL_WRAP_WIDTH = 11
# Explicit line breaks for labels where generic wrapping doesn't give the
# desired split.
LABEL_OVERRIDES = {
    "Longitudinal tumour volume": "Longitudinal\ntumour volume",
    "Analysis-ready dataset": "Analysis-ready\ndataset",
}
# Short forms for the figure's row-label subtitle -- distinct from the fuller
# text in Sheet3's "Organization" column, which is too long to fit under a
# resource name without repeating the header-collision problem long labels
# already caused once. Keyed by exact "PDX Resource" cell text -- a row rename
# needs a matching update here (same tradeoff as LABEL_OVERRIDES above).
ORG_SHORT_LABELS = {
    "PDX Finder / PDCM Finder (retired)": "EMBL-EBI / JAX",
    "EurOPDX": "European consortium",
    "PDMR": "NCI",
    "PDXNet": "NCI",
    "XevaDB": "PMCC",
}
# Line breaks for resource names too long to sit on one line -- same
# rationale/tradeoff as LABEL_OVERRIDES above (keyed by exact "PDX Resource"
# cell text; a rename needs a matching update here).
RESOURCE_LABEL_OVERRIDES = {
    "PDX Finder / PDCM Finder (retired)": "PDX Finder /\nPDCM Finder (retired)",
}
COLUMN_SHADE_COLORS = {
    "Model metadata": "#D6EAC4",
    "Molecular profiles": "#FCEEA6",
    "Drug response": "#CDE7F4",
    "Longitudinal tumour volume": "#7FB8D9",
    "Analysis-ready dataset": "#F5C6C6",
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

    fig, ax = plt.subplots(figsize=(2.0 * n_cols + 2, 0.6 * n_rows + 2.2))

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

    # Two-tier row labels (resource name + smaller/greyed organization
    # subtitle) instead of plain yticklabels, since a single Text artist
    # can't mix font sizes across its lines.
    ax.set_yticklabels([])
    for row_i, resource in enumerate(resources):
        org = ORG_SHORT_LABELS.get(resource)
        display_name = RESOURCE_LABEL_OVERRIDES.get(resource, resource)
        two_line_name = "\n" in display_name
        # The name's vertical CENTER always sits on row_i, exactly where its
        # dots are -- that alignment takes priority. The org subtitle just
        # hangs below it, pushed further down when the name itself is 2
        # lines tall so it clears the name's bottom edge.
        name_y = row_i
        org_y = row_i + (0.43 if two_line_name else 0.30) if org else None
        ax.text(
            -0.6, name_y, display_name, transform=ax.transData, ha="right", va="center",
            fontsize=plt.rcParams["ytick.labelsize"],
            fontweight="bold" if resource == BOLD_RESOURCE else "normal",
            clip_on=False,
        )
        if org:
            ax.text(
                -0.6, org_y, org, transform=ax.transData, ha="right", va="center",
                fontsize=plt.rcParams["ytick.labelsize"] * 0.95, color="#666666",
                clip_on=False,
            )

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
