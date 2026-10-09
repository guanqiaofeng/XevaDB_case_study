# results/

Intermediate and final outputs of each case study's own pipeline — fold-level
model predictions, QC tables, fitted models, and NeST-VNN input matrices.
Finished manuscript figures/tables live in `../figures_tables/` instead;
everything here is either an input to that final step or a QC/diagnostic
artifact documented in the owning case study's own README.

| Folder | Case study | Notable subfolders |
|---|---|---|
| `1_doublingRate/` | 1 | flat — doubling-time fits, ML fold results, GSEA rankings, QC tables (breast/lung suffixed) |
| `2_drugSensitivity/` | 2 | `expert_rating/` (consensus labels + CV folds), `images_clean/` (1,300 cropped PNGs), `ml_model/{resnet18,random_forest}/` (fold checkpoints, predictions, SHAP) |
| `3_carboplatin/` | 3 | `McGill_nest_input/`, `UHN_nest_input/{complete,strict}_cohort/` — NeST-VNN-format input matrices (`cell2ind.txt`, `cell2mutation.txt`, etc.) |
| `4_paclitaxel/` | 4 | flat — processed omics matrices, classification fold results/predictions, GSEA tables |

Each case folder's own `pipeline_utils.py` resolves these paths via
`make_case{N}_paths()["results"]` (or `["proc"]`, the same directory) —
regenerate by rerunning that case's notebooks in order rather than
hand-editing anything here.
