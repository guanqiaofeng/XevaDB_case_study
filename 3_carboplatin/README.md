# 3_carboplatin/ — Case study 3: cross-platform NeST-VNN validation

Runs a cell-line-pretrained NeST-VNN ensemble (cisplatin-trained) on two
independent breast PDX cohorts it was never trained on — McGill (cisplatin,
n=29) and UHN (carboplatin; complete n=68, replicate-supported "strict"
n=36) — and validates its predictions against each cohort's own
growth-curve response label. Maps to **Figure 4** and **Supplemental
Figure 3**.

- **Environment**: `../requirements.txt` for the notebooks; the modeling
  step also needs the separate `pretrained_models/nest_vnn/nest_env/`
  virtualenv (see that folder's own `read_me_to_run.txt`) and R
  (`recompute_sensitivity_min10_max28.R`, `window_confound_sweep.R`) with
  the `Xeva` package.
- **Inputs**: `../procdata/mcgill_breast`, `../procdata/uhn_breast`, and
  the pretrained ensemble in `../pretrained_models/nest_vnn/`.
- **Outputs**: `../results/3_carboplatin` and `../figures_tables/`.

## Run order

| Notebook/script | Purpose | Key outputs |
|---|---|---|
| `window_confound_sweep.R` | Diagnoses whether/how the response window should be chosen (does follow-up duration confound slope?) — run once, informs the fixed choice below | `results/3_carboplatin/{cohort}_slope_{full_duration,sweep}.csv`; Supp Fig 3b |
| `recompute_sensitivity_min10_max28.R` | Recomputes slope/angle/mRECIST for both cohorts on a common, explicit 10–28 day window | windowed batch/model sensitivity tables consumed by steps 1–2 |
| `1-McGill_dataset_preprocessing.ipynb` | Filter to cisplatin/carboplatin experiments, map to CNV-bearing omics samples, build NeST-VNN input matrices | `results/3_carboplatin/McGill_nest_input/` |
| `2-UHN_dataset_preprocessing.ipynb` | Same for UHN, builds both the complete and strict cohorts | `results/3_carboplatin/UHN_nest_input/{complete,strict}_cohort/` |
| `3-cohort_comparison.ipynb` | Gene-coverage, event-burden, and PCA comparability checks between cohorts; re-derives and applies the chosen response window | Fig 4b/c, Supp Fig 3a/c/d/e, **Table 8**, **Table 9**, **Table 10** |
| `4-nestvnn_modeling.ipynb` | Runs the pretrained 5-fold cisplatin NeST-VNN ensemble (`predict.py`, via the `nest_env` subprocess) on each cohort's prepared input | per-cohort, per-fold prediction CSVs |
| `5-result_interpretation.ipynb` | Aggregates the ensemble predictions; bootstrap AUROC at the fixed threshold, percentile-cutoff robustness sweep, capped-vs-uncapped comparison | Fig 4d/e, Supp Fig 3f, **Table 5**, **Table 6**, **Table 7** |

The two `recompute_sensitivity_*`/`window_confound_sweep.R` scripts must run
before `1`/`2`; `1` and `2` are independent of each other; `3` needs both;
`4` needs `3`; `5` needs `4`.

## Notes

- **Response window (10–28 days)** is fixed by `window_confound_sweep.R`'s
  own diagnostic — the deepest (most extreme median response) point in a
  14–56 day sweep — not by whichever window maximizes downstream AUROC.
  Case study 4 reuses this exact methodology (its own sweep, different
  drug) but not this window's value.
- **TGI is dropped**, not recomputed, for the same reason as case study 2:
  `Xeva::TGI()` errors on any batch with zero in-window data points on one
  arm within the chosen window.
- `4-nestvnn_modeling.ipynb` only ever uses the **cisplatin** pretrained
  ensemble — the other 7 drug-specific ensembles bundled in
  `pretrained_models/nest_vnn/pretrained_models/` are gitignored and unused
  here.
- Bootstrap CIs/p-values in step 5 are seeded (`seed=0`), so a rerun should
  reproduce exactly — any drift traces back to the NeST-VNN ensemble
  inference step (`4-nestvnn_modeling.ipynb`) upstream, not the bootstrap
  itself.
