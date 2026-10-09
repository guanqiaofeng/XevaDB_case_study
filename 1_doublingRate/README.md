# 1_doublingRate/ — Case study 1: transcriptomic modeling of intrinsic growth kinetics

Estimates model-level doubling time from control-arm tumour-volume curves in
the McGill/UHN breast and UHN lung cohorts, then asks whether RNA-seq can
classify the fastest- vs. slowest-growing tertile (ElasticNet, Random
Forest, Lassoed Forest; repeated CV). Maps to **Figure 2** and
**Supplemental Figure 1**.

Environment: `../requirements.txt`. `compute_limma_ranking.R` needs R with
`limma`.

## Run order

| Script | Purpose | Key outputs |
|---|---|---|
| `1-breast_DT_preprocessing.ipynb` | QC + early-phase log-linear growth fit per control-arm mouse/model, breast | `results/1_doublingRate/breast_doubling_time.csv`; Fig 2b (representative fit), Supp Fig 1a/b |
| `1-lung_DT_preprocessing.ipynb` | Same, lung | lung equivalents |
| `2-breast_rna_preprocessing.ipynb` | Filter to protein-coding genes, merge RNA-seq with doubling time | Fig 2c (breast) |
| `2-lung_rna_preprocessing.ipynb` | Same, lung | Fig 2c (lung) |
| `3-breast_ml_modeling.py` | Fastest/slowest-tertile classification (ElasticNet/RF/LassoedForest, nested CV), breast — thin wrapper around `pipeline_utils.run_growth_ml_pipeline` | `growth_ml_config_breast.json`, `growth_lassoedforest_sensitivity_fold_results_breast.csv`, etc. |
| `3-lung_ml_modeling.py` | Same, lung | lung equivalents |
| `4-ml_model_interpretation.ipynb` | Aggregates both cohorts: performance comparison, Hallmark GSEA (via `compute_limma_ranking.R`) | Fig 2d, Fig 2e, **Table 1**, **Table 2** |

Run per cohort in the `1 → 2 → 3` order shown (breast and lung are
independent of each other); `4` needs both cohorts' step-3 output.

`compute_limma_ranking.R` is a generic two-group limma moderated-t helper
(also used by case study 4) — not run standalone, called via `subprocess`
from `4-ml_model_interpretation.ipynb`.

## `supp/` — N_HVG sensitivity justification

Documents why `N_HVG` (the highly-variable-gene count fed to the
classifiers) is fixed at a different, cohort-specific value for breast
(100) vs. lung (600), applied identically across all three models rather
than tuned per model. `1-lassoedforest_hvg_grid_{breast,lung}.py` run the
`N_HVG × max_depth × min_samples_leaf` grid search feeding
`2-hvg_selection_justification.ipynb`'s evidence plots → **Supplemental
Figure 1c/d**.

## Notes

- `pipeline_utils.py`'s `run_growth_ml_pipeline()` is the shared
  implementation behind both `3-*_ml_modeling.py` scripts; most
  `results/1_doublingRate/*.csv` filenames are built dynamically from its
  `cohort` argument (`f"growth_ml_config_{cohort}.json"` etc.), not literal
  strings in the calling scripts.
- MAX_DAY (35-day follow-up cutoff), MIN_POINTS, and MIN_R2 are justified by
  the confound/sensitivity diagnostics in step 1's notebooks (Supp Fig 1a/b)
  the same way case studies 3 and 4 justify their own response windows —
  characterizing the data's own confound structure, not maximizing
  downstream performance.
