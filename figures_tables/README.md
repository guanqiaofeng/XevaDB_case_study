# figures_tables/

Manuscript-ready figures and tables. The three root files are the
submission masters; `figures_main/`, `figures_supp/`, and `tables/` hold the
individual panel/table files each master is assembled from.

| File | Contents |
|---|---|
| `Figures.pdf` / `Figures.pptx` | Main-text Figures 1–5 |
| `Supp_Figures.pdf` / `Supp_Figures.pptx` | Supplemental Figures 1–4 |
| `Supp_Tables.xlsx` | Supplementary Tables 1–14, one sheet each |

Panel (a) of every main figure (and panel (f) of Figure 1) is a schematic
workflow diagram made in BioRender — it has no generating script and no
case study owns it (Figure 1's panels are a cross-cohort summary, not tied
to one case study).

## `figures_main/`

| File | Case study | Generated from |
|---|---|---|
| `figure1a_study_summary.png` | Summary | BioRender |
| `figure1b_summary_counts.pdf/png` | Summary | `0_xevaset_extraction/7-figure1_panels.ipynb` |
| `figure1c_cancer_type_distribution.pdf/png` | Summary | `0_xevaset_extraction/7-figure1_panels.ipynb` |
| `figure1d_shared_drugs.pdf/png` | Summary | `0_xevaset_extraction/7-figure1_panels.ipynb` |
| `figure1e_resource_comparison.pdf/png` | Summary | `0_xevaset_extraction/7-figure1_panels.ipynb` |
| `figure1f_XevaSet_structure.png` | Summary | BioRender |
| `figure2a_case1_summary.png` | Case 1 | BioRender |
| `figure2b_growth_kinetics_breast.pdf/png` | Case 1 | `1_doublingRate/1-breast_DT_preprocessing.ipynb` |
| `figure2c_DT_breast.pdf/png` | Case 1 | `1_doublingRate/2-breast_rna_preprocessing.ipynb` |
| `figure2c_DT_lung.pdf/png` | Case 1 | `1_doublingRate/2-lung_rna_preprocessing.ipynb` |
| `figure2d_Model_Comparison.pdf/png` | Case 1 | `1_doublingRate/4-ml_model_interpretation.ipynb` |
| `figure2e_Main_Shared_Hallmark_NES.pdf/png` | Case 1 | `1_doublingRate/4-ml_model_interpretation.ipynb` |
| `figure3a_case2_summary.png` | Case 2 | BioRender |
| `figure3b_model_comparison.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| `figure3c_confusion_matrices.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| `figure3d_gradcam.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| `figure4a_case3_summary.png` | Case 3 | BioRender |
| `figure4b_replicate_counts.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `figure4c_gene_coverage.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `figure4d_capped_vs_uncapped_auroc.pdf/png` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| `figure4e_auroc_percentile_sweep.pdf/png` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| `figure5_case4_summary.png` | Case 4 | BioRender |
| `figure5b_model_benchmark.pdf/png` | Case 4 | `4_paclitaxel/3-figures.ipynb` |
| `figure5c_gsea_hallmark.pdf/png` | Case 4 | `4_paclitaxel/3-figures.ipynb` |
| `figure5d_ispy2_validation.pdf/png` | Case 4 | `4_paclitaxel/4-ispy2_validation.ipynb` |

## `figures_supp/`

| File | Case study | Generated from |
|---|---|---|
| `supp_figure1a_qc_threshold_sensitivity_breast.pdf/png` | Case 1 | `1_doublingRate/1-breast_DT_preprocessing.ipynb` |
| `supp_figure1a_qc_threshold_sensitivity_lung.pdf/png` | Case 1 | `1_doublingRate/1-lung_DT_preprocessing.ipynb` |
| `supp_figure1b_followup_censoring_breast.pdf/png` | Case 1 | `1_doublingRate/1-breast_DT_preprocessing.ipynb` |
| `supp_figure1b_followup_censoring_lung.pdf/png` | Case 1 | `1_doublingRate/1-lung_DT_preprocessing.ipynb` |
| `supp_figure1c_hvg_generalization.pdf/png` | Case 1 | `1_doublingRate/supp/2-hvg_selection_justification.ipynb` |
| `supp_figure1d_hvg_lassoedforest_grid.pdf/png` | Case 1 | `1_doublingRate/supp/2-hvg_selection_justification.ipynb` |
| `supp_figure2c_resnet18_training_curves.pdf/png` | Case 2 | `2_drugSensitivity/3-ml_resnet18.ipynb` |
| `supp_figure2d_per_class_comparison.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| `supp_figure2e_rf_shap_feature_importance.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| `supp_figure3a_event_burden_density.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `supp_figure3a_mutation_coverage_sensitivity.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `supp_figure3b_window_confound_diagnostic.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `supp_figure3c_pca.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `supp_figure3d_control_arm_ks.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `supp_figure3e_treatment_arm_slope.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `supp_figure3f_roc_curves.pdf/png` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| `supp_figure4a_window_confound_diagnostic.pdf/png` | Case 4 | `4_paclitaxel/1-data_preprocessing.ipynb` |
| `supp_figure4b_response_endpoint_qc.pdf/png` | Case 4 | `4_paclitaxel/3-figures.ipynb` |
| `supp_figure4c_gsea_full_detail.pdf/png` | Case 4 | `4_paclitaxel/3-figures.ipynb` |

## `tables/`

| File | Case study | Generated from |
|---|---|---|
| `table1_case1_ml_performance_summary.csv` | Case 1 | `1_doublingRate/4-ml_model_interpretation.ipynb` |
| `table2_case1_hallmark_gsea_concordant.csv` | Case 1 | `1_doublingRate/4-ml_model_interpretation.ipynb` |
| `table3_case2_ml_model_performance.csv` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| `table4_case2_drug_response_per_class_performance.csv` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| `table5_case3_nestvnn_model_performance.csv` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| `table6_case3_capped_vs_uncapped_auroc.csv` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| `table7_case3_auroc_threshold_sensitivity.csv` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| `table8_case3_nestvnn_gene_coverage.csv` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `table9_case3_mutation_cna_cnd_event_burden.csv` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `table10_case3_growth_curve_slope_ks.csv` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| `table12_case4_classification_model_benchmark.csv` | Case 4 | `4_paclitaxel/2-ML.ipynb` |
| `table13_case4_ispy2_ctr_arm_validation.csv` | Case 4 | `4_paclitaxel/4-ispy2_validation.ipynb` |

Two sheets in `Supp_Tables.xlsx` have no CSV here because they aren't
pipeline-generated outputs copied into this folder:
**Table 11** (curated paclitaxel gene list) mirrors
`results/4_paclitaxel/pax_gene_list.tsv`, a hand-curated list (its
`Function` column is written by hand, not generated). **Table 14** (AIMS
subtypes) mirrors `results/4_paclitaxel/uhn_breast_pam50_aims_subtype.csv`,
generated by `4_paclitaxel/1-data_preprocessing.ipynb` via
`compute_pam50_subtype_aims.R` — a real case study 4 output, just not
copied into `tables/`.

## Regenerating

Each panel/table file is written by the notebook listed above, by path
(`paths["figures"]`, `paths["figures_supp"]`, or `paths["tables"]` from that
case's `pipeline_utils.py`) — rerun that case study's notebooks in order to
regenerate it. The three root masters (`Figures.pdf` etc.) and every
BioRender panel are assembled/drawn externally and are not generated by any
script in this repo — re-embed updated panels by hand after a rerun.
