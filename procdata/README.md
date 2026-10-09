# procdata/

Flat, per-cohort CSV (and for PDXE, image) tables extracted from the raw
XevaSet `.rds` objects in `rawdata/` by `0_xevaset_extraction/`. Every case
study reads from here, never from `rawdata/` directly.

| Folder | Built by | Contents |
|---|---|---|
| `mcgill_breast/` | `0_xevaset_extraction/3-mcgill_breast_extraction.R` | experiment/expDesign/drug/models/modToBiobaseMap, RNA-seq/CNV/mutation/fusion/segment matrices + gene annotations, batch- and model-level sensitivity tables |
| `uhn_breast/` | `4-uhn_breast_extraction.R` | same shape as `mcgill_breast/` (no fusion/segment matrices) |
| `uhn_lung/` | `5-uhn_lung_extraction.R` | same shape, no sensitivity tables (UHN lung is used for doubling-time only, not drug response) |
| `pdxe/csv/` | `1-pdxe_tables_extraction.R` | models/expDesign/modToBiobaseMap/experiment/drug core tables |
| `pdxe/images/` | `2-pdxe_image_manifest_extraction.R` | 1,300 rendered tumour-volume images (`*.webp`) for case study 2's image classifier, plus their response-metric `manifest.csv` and `failed_batches.txt` |
| `summary/` | `6-build_dataset_summary.py` (+ one hand-curated file) | `dataset_summary.xlsx` (Sheet1 counts, Sheet2 cancer-type breakdown), `drug_dataset_overlap.csv` — both feed Figure 1b–d. `database_summary.xlsx` is a **separate, hand-curated** XevaDB-vs.-external-resources comparison feeding Figure 1e; it has no generating script. |

## Notes

- Column naming is inconsistent *between* cohorts in a few places
  (`RNAseq_matrix.csv` vs. `RNASeq_matrix.csv`, `CNV_gene_annotation.csv`
  vs. `cnv_gene_annotation.csv`) — this mirrors the source XevaSets'
  own inconsistency and is handled case-by-case in each case study's
  loading code rather than normalized here.
- `pdxe/` splits into `csv/` (tabular) and `images/` (webp + manifest +
  failed-batch log) — case study 2's notebooks read both from
  `paths["datasets"] / "images"`.
- Nothing in this folder should be hand-edited; regenerate via
  `0_xevaset_extraction/` instead.
