# 4_paclitaxel/ — Case study 4: multi-omics prediction of paclitaxel response

Predicts paclitaxel response (fixed `slope.treatment < 0` cutoff) in the
UHN breast PDX cohort from RNA-seq/CNV/mutation, comparing single-omics,
early-fusion, and late-fusion (stacking) classifiers, then externally
validates the top RNA pathway findings against the I-SPY2 clinical trial's
paclitaxel control arm. Maps to **Figure 5** and **Supplemental Figure 4**.

Environment: `../requirements.txt`, plus R (`Xeva`, `limma`) for the
helper scripts and `gseapy` for GSEA.

## Run order

| Notebook/script | Purpose | Key outputs |
|---|---|---|
| `window_confound_sweep.R` | Diagnoses the response window the same way as case study 3 (own sweep, this drug's data) | Supp Fig 4a |
| `recompute_sensitivity_min10_max49.R` | Recomputes slope/angle/mRECIST on the chosen 10–49 day window; called via `subprocess` from `1-data_preprocessing.ipynb`, not run standalone | windowed batch/model sensitivity tables |
| `1-data_preprocessing.ipynb` | Extracts RNA-seq/CNV/mutation, applies the response window, defines the binary endpoint, runs AIMS subtyping | `results/4_paclitaxel/{rna_logtpm,cnv,mutation,metadata}.csv`; Supp Fig 4a/b, **Table 14** |
| `2-ML.ipynb` | Fold-safe feature selection + single-omics / early-fusion / late-fusion stacking classification, repeated CV (Random Forest, Logistic L2) | `results/4_paclitaxel/*_classification_cv_{results,predictions}.csv`, **Table 12** |
| `3-figures.ipynb` | Builds Fig 5b/c and Supp Fig 4b/c from `2-ML.ipynb`'s saved outputs; does not rerun model training | Fig 5b/c, Supp Fig 4b/c |
| `4-ispy2_validation.ipynb` | Refits the I-SPY2 Ctr-arm model to recover a 95% CI, cross-checks signature gene overlap, builds the PDX-vs-I-SPY2 comparison | Fig 5d, **Table 13** |

`compute_pam50_subtype_aims.R` (AIMS intrinsic molecular subtyping) and
`compute_limma_ranking.R` (shared with case study 1) are called via
`subprocess` from the notebooks above, not run standalone.
`feature_sets.py` holds the fold-safe feature-selection helpers shared by
`2-ML.ipynb` and `3-figures.ipynb`.

Order: `window_confound_sweep.R` informs the window choice baked into
`recompute_sensitivity_min10_max49.R`, called from `1`; then `1 → 2 → 3`;
`4` only needs `2`'s GSEA output (via `3`'s GSEA cell) and the raw
`rawdata/I-SPY2_clinical_trial/` files.

## Notes

- **Primary endpoint**: fixed `slope.treatment < 0` cutoff (Sensitive) vs.
  `>= 0` (Resistant) — matches case study 3's convention exactly, not a
  median/distribution-derived split. mRECIST is kept only as a reference/QC
  column (`feature_sets.add_binary_response`).
- **Multiple-comparisons correction**: Benjamini-Hochberg FDR across all 10
  model/omics/analysis combinations tested in `2-ML.ipynb` (5 categories ×
  2 models, collapsing early/late fusion's single category each).
- **AIMS, not genefu PAM50**, for intrinsic subtyping: there is no
  clinical ER/PR/HER2 annotation anywhere in the UHN breast XevaSet, and
  AIMS is a single-sample rank-based classifier that doesn't need a
  reference cohort to normalize against — a better fit for PDX RNA-seq
  with no matched normal panel.
- I-SPY2 source files (`rawdata/I-SPY2_clinical_trial/mmc{2,3,4}.xlsx`) are
  saved as Strict Open XML Spreadsheets; `4-ispy2_validation.ipynb`'s
  `read_strict_ooxml_excel()` patches the namespace in-memory before
  handing the bytes to pandas — the files on disk are never modified.
