# procdata/

Flat, per-cohort CSV (and for PDXE, image) tables extracted from the raw
XevaSet `.rds` objects in `rawdata/` by `0_xevaset_extraction/`. Every case
study reads from here.

| Folder | Built by | Contents |
|---|---|---|
| `pdxe/csv/` | `1-pdxe_tables_extraction.R` | models/expDesign/modToBiobaseMap/experiment/drug core tables |
| `pdxe/images/` | `2-pdxe_image_manifest_extraction.R` | 1,300 rendered tumour-volume images (`*.webp`) for case study 2's image classifier, plus their response-metric `manifest.csv` and `failed_batches.txt` |
| `mcgill_breast/` | `3-mcgill_breast_extraction.R` | experiment/expDesign/drug/models/modToBiobaseMap, RNA-seq/CNV/mutation/fusion/segment matrices + gene annotations, batch- and model-level sensitivity tables |
| `uhn_breast/` | `4-uhn_breast_extraction.R` | experiment/expDesign/drug/models/modToBiobaseMap, RNA-seq/CNV/mutation + gene annotations, batch- and model-level sensitivity tables |
| `uhn_lung/` | `5-uhn_lung_extraction.R` | experiment/expDesign/drug/models/modToBiobaseMap, RNA-seq/CNV/mutation + gene annotations |
| `summary/` | `6-build_dataset_summary.py` | `dataset_summary.xlsx` , `drug_dataset_overlap.csv` — both feed Figure 1b–d. |


