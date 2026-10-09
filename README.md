# XevaDB_case_study

Reproducible pipeline for four PDX pharmacogenomic case studies built on
**XevaDB**, a standardized resource integrating four PDX cohorts (McGill
breast, UHN breast, UHN lung, PDXE pan-cancer): growth-kinetics modeling,
image-based drug-response classification, cell-line-to-PDX transfer
learning, and multi-omics paclitaxel response prediction.

## Repository layout

```
rawdata/              Raw, unmodified source XevaSets + external data — see rawdata/README.md
procdata/              Flat per-cohort tables extracted from rawdata/   — see procdata/README.md
0_xevaset_extraction/  Extraction scripts: rawdata/ -> procdata/, + Figure 1
1_doublingRate/        Case study 1: growth-kinetics modeling         (Figure 2, Supp Fig 1)
2_drugSensitivity/      Case study 2: image-based response classification (Figure 3, Supp Fig 2)
3_carboplatin/          Case study 3: cross-platform NeST-VNN validation   (Figure 4, Supp Fig 3)
4_paclitaxel/           Case study 4: multi-omics paclitaxel prediction    (Figure 5, Supp Fig 4)
results/                Per-case intermediate/final pipeline outputs    — see results/README.md
figures_tables/         Manuscript figures/tables (submission masters) — see figures_tables/README.md
pretrained_models/      Pretrained NeST-VNN ensemble (external, used by case study 3)
```

Each of `1_doublingRate/` … `4_paclitaxel/` has its own README with the
exact notebook run order and figure/table mapping.

## Environment setup

Two separate environments are required:

- **`requirements.txt`** — everything except the ResNet18 notebook.
  Verified against Python 3.9.6.
- **`requirements-resnet.txt`** — `2_drugSensitivity/3-ml_resnet18.ipynb`
  only (newer `torch`/`torchvision`/core-package versions; keep in its own
  kernel, do not install alongside `requirements.txt`).

R scripts (data extraction, case studies 1/3/4's response-window helpers)
need R with the `Xeva` package, plus `limma` (GSEA ranking) and `dplyr`/
`ggplot2`.

`3_carboplatin/4-nestvnn_modeling.ipynb` additionally needs the pretrained
NeST-VNN ensemble's own virtualenv — see
`pretrained_models/nest_vnn/read_me_to_run.txt`.

## Reproduction order

1. **`0_xevaset_extraction/`** — must run first; every case study reads
   from `procdata/`, not `rawdata/`.
2. **`1_doublingRate/`, `2_drugSensitivity/`, `3_carboplatin/`,
   `4_paclitaxel/`** — independent of each other and of execution order;
   each reads only from `procdata/`/`rawdata/` and its own case folder.
   Within each, follow that folder's own README for notebook order.

Final figures/tables are read from `results/` into `figures_tables/` by
each case study's own later notebooks (not a separate aggregation step).

## Case study summary

| # | Folder | Question | Primary model |
|---|---|---|---|
| 1 | `1_doublingRate/` | Does RNA-seq predict intrinsic PDX growth rate? | ElasticNet / Random Forest / Lassoed Forest |
| 2 | `2_drugSensitivity/` | Can tumour-volume curve images classify drug response as well as engineered features? | ResNet18 vs. Random Forest |
| 3 | `3_carboplatin/` | Does a cell-line-pretrained NeST-VNN ensemble transfer to independent PDX cohorts? | Pretrained NeST-VNN (cisplatin) |
| 4 | `4_paclitaxel/` | Does multi-omics fusion improve paclitaxel response prediction, and does it validate externally? | Random Forest / Logistic L2, single/early/late fusion |
