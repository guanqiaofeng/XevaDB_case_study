# figures_tables/

Including all figures and tables of the study. 

Panel (a) of every main figure (and panel (f) of Figure 1) is a schematic
workflow diagram made in BioRender. Supp table 11 is a customized gene list; and Supp fig 2a is a customized rating criteria. 

ALL other figures and tables are generated from the workflow. See details below.

## `figures_main/`

| Figure ID | File | Case study | Generated from |
|---|---|---|---|
| Fig 1a | `figure1a_study_summary.png` | Summary | BioRender |
| Fig 1b | `figure1b_summary_counts.pdf/png` | Summary | `0_xevaset_extraction/7-figure1_panels.ipynb` |
| Fig 1c | `figure1c_cancer_type_distribution.pdf/png` | Summary | `0_xevaset_extraction/7-figure1_panels.ipynb` |
| Fig 1d | `figure1d_shared_drugs.pdf/png` | Summary | `0_xevaset_extraction/7-figure1_panels.ipynb` |
| Fig 1e | `figure1e_resource_comparison.pdf/png` | Summary | `0_xevaset_extraction/7-figure1_panels.ipynb` |
| Fig 1f | `figure1f_XevaSet_structure.png` | Summary | BioRender |
| Fig 2a | `figure2a_case1_summary.png` | Case 1 | BioRender |
| Fig 2b | `figure2b_growth_kinetics_breast.pdf/png` | Case 1 | `1_doublingRate/1-breast_DT_preprocessing.ipynb` |
| Fig 2c | `figure2c_DT_breast.pdf/png` | Case 1 | `1_doublingRate/2-breast_rna_preprocessing.ipynb` |
| Fig 2c | `figure2c_DT_lung.pdf/png` | Case 1 | `1_doublingRate/2-lung_rna_preprocessing.ipynb` |
| Fig 2d | `figure2d_Model_Comparison.pdf/png` | Case 1 | `1_doublingRate/4-ml_model_interpretation.ipynb` |
| Fig 2e | `figure2e_Main_Shared_Hallmark_NES.pdf/png` | Case 1 | `1_doublingRate/4-ml_model_interpretation.ipynb` |
| Fig 3a | `figure3a_case2_summary.png` | Case 2 | BioRender |
| Fig 3b | `figure3b_model_comparison.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| Fig 3c | `figure3c_confusion_matrices.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| Fig 3d | `figure3d_gradcam.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| Fig 4a | `figure4a_case3_summary.png` | Case 3 | BioRender |
| Fig 4b | `figure4b_replicate_counts.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Fig 4c | `figure4c_gene_coverage.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Fig 4d | `figure4d_capped_vs_uncapped_auroc.pdf/png` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| Fig 4e | `figure4e_auroc_percentile_sweep.pdf/png` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| Fig 5a | `figure5_case4_summary.png` | Case 4 | BioRender |
| Fig 5b | `figure5b_model_benchmark.pdf/png` | Case 4 | `4_paclitaxel/3-figures.ipynb` |
| Fig 5c | `figure5c_gsea_hallmark.pdf/png` | Case 4 | `4_paclitaxel/3-figures.ipynb` |
| Fig 5d | `figure5d_ispy2_validation.pdf/png` | Case 4 | `4_paclitaxel/4-ispy2_validation.ipynb` |

## `figures_supp/`

| Supp Figure ID | File | Case study | Generated from |
|---|---|---|---|
| Supp Fig 1a | `supp_figure1a_qc_threshold_sensitivity_breast.pdf/png` | Case 1 | `1_doublingRate/1-breast_DT_preprocessing.ipynb` |
| Supp Fig 1a | `supp_figure1a_qc_threshold_sensitivity_lung.pdf/png` | Case 1 | `1_doublingRate/1-lung_DT_preprocessing.ipynb` |
| Supp Fig 1b | `supp_figure1b_followup_censoring_breast.pdf/png` | Case 1 | `1_doublingRate/1-breast_DT_preprocessing.ipynb` |
| Supp Fig 1b | `supp_figure1b_followup_censoring_lung.pdf/png` | Case 1 | `1_doublingRate/1-lung_DT_preprocessing.ipynb` |
| Supp Fig 1c | `supp_figure1c_hvg_generalization.pdf/png` | Case 1 | `1_doublingRate/supp/2-hvg_selection_justification.ipynb` |
| Supp Fig 1d | `supp_figure1d_hvg_lassoedforest_grid.pdf/png` | Case 1 | `1_doublingRate/supp/2-hvg_selection_justification.ipynb` |
| Supp Fig 2a | `supp_figure2a_panel_of_expert_rating_criteria.png` | Case 2 | customized rating criteria |
| Supp Fig 2b | `supp_figure2b_panel_of_expert_consensus.png` | Case 2 | `2_drugSensitivity/1-expertRating_preprocessing.ipynb` |
| Supp Fig 2c | `supp_figure2c_resnet18_training_curves.pdf/png` | Case 2 | `2_drugSensitivity/3-ml_resnet18.ipynb` |
| Supp Fig 2d | `supp_figure2d_per_class_comparison.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| Supp Fig 2e | `supp_figure2e_rf_shap_feature_importance.pdf/png` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| Supp Fig 3a | `supp_figure3a_event_burden_density.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Fig 3a | `supp_figure3a_mutation_coverage_sensitivity.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Fig 3b | `supp_figure3b_window_confound_diagnostic.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Fig 3c | `supp_figure3c_pca.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Fig 3d | `supp_figure3d_control_arm_ks.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Fig 3e | `supp_figure3e_treatment_arm_slope.pdf/png` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Fig 3f | `supp_figure3f_roc_curves.pdf/png` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| Supp Fig 4a | `supp_figure4a_window_confound_diagnostic.pdf/png` | Case 4 | `4_paclitaxel/1-data_preprocessing.ipynb` |
| Supp Fig 4b | `supp_figure4b_response_endpoint_qc.pdf/png` | Case 4 | `4_paclitaxel/3-figures.ipynb` |
| Supp Fig 4c | `supp_figure4c_gsea_full_detail.pdf/png` | Case 4 | `4_paclitaxel/3-figures.ipynb` |

## `tables/`

| Supp Table ID | File | Case study | Generated from |
|---|---|---|---|
| Supp Table 1 | `table1_case1_ml_performance_summary.csv` | Case 1 | `1_doublingRate/4-ml_model_interpretation.ipynb` |
| Supp Table 2 | `table2_case1_hallmark_gsea_concordant.csv` | Case 1 | `1_doublingRate/4-ml_model_interpretation.ipynb` |
| Supp Table 3 | `table3_case2_ml_model_performance.csv` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| Supp Table 4 | `table4_case2_drug_response_per_class_performance.csv` | Case 2 | `2_drugSensitivity/5-ml_model_interpretation.ipynb` |
| Supp Table 5 | `table5_case3_nestvnn_model_performance.csv` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| Supp Table 6 | `table6_case3_capped_vs_uncapped_auroc.csv` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| Supp Table 7 | `table7_case3_auroc_threshold_sensitivity.csv` | Case 3 | `3_carboplatin/5-result_interpretation.ipynb` |
| Supp Table 8 | `table8_case3_nestvnn_gene_coverage.csv` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Table 9 | `table9_case3_mutation_cna_cnd_event_burden.csv` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Table 10 | `table10_case3_growth_curve_slope_ks.csv` | Case 3 | `3_carboplatin/3-cohort_comparison.ipynb` |
| Supp Table 11 | `table11_case4_paclitaxel_related_gene_list.tsv` | Case 3 | customized gene list |
| Supp Table 12 | `table12_case4_classification_model_benchmark.csv` | Case 4 | `4_paclitaxel/2-ML.ipynb` |
| Supp Table 13 | `table13_case4_ispy2_ctr_arm_validation.csv` | Case 4 | `4_paclitaxel/4-ispy2_validation.ipynb` |
| Supp Table 14 | `table14_case4_uhn_breast_pam50_aims_subtype.csv` | Case 4 | `4_paclitaxel/1-data_preprocessing.ipynb` (via `compute_pam50_subtype_aims.R`) |

