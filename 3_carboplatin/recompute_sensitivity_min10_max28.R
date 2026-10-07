### Recompute drug-sensitivity metrics for the McGill and UHN breast XevaSets
# using a common, explicit 10-28 day response window (min.time = 10,
# max.time = 28), so both cohorts are comparable on the same measurement
# basis for case study 3: slope + angle at the batch level, mRECIST at the
# model level (batch-level mRECIST is then derived downstream by majority
# vote across each batch's treatment-arm mice, ties broken toward the more
# resistant call -- same as before, just fed freshly-windowed mRECIST calls).
#
# min.time only affects Xeva's mRECIST() (it requires the best-response
# search and the data-sufficiency check to use time >= min.time); it has
# ZERO effect on slope()/angle() -- verified empirically (slope.treatment
# identical to floating-point noise regardless of min.time on this data).
# min.time=10 is kept deliberately: min.time=0 was tested and rejected, since
# it makes day 0 itself (baseline vs. itself, trivially 0% change) a valid
# "best response" candidate, which structurally makes the PD category
# unreachable for every model regardless of actual growth -- an artifact, not
# a real finding.
#
# Window choice (max.time=28): see window_confound_sweep.R / the
# "Window-choice justification" cells in 3-cohort_comparison.ipynb. McGill's
# median treatment-arm slope, swept over candidate max.time in weekly steps,
# bottoms out (deepest, least regrowth-contaminated response) at max.time=28
# and erodes back toward "non-responding" at wider windows -- 28 is the
# data's own inflection point, not a value chosen to maximize downstream
# AUROC. (McGill's original batch_sensitivity.csv from 0_xevaset_extraction_summary/
# used max.time=30 instead, a value inherited from however that RDS was
# originally built, not derived from this diagnostic.)
#
# UHN's original batch_sensitivity.csv reflects the full observed treatment
# duration instead (median 40.5 days, up to 286 days), and shows little
# preference either way across the max.time sweep. Recomputing both here --
# rather than editing 0_xevaset_extraction_summary/, whose UHN output is also used
# unmodified by case study 4 (paclitaxel) -- keeps this a case-3-local choice.
#
# Output (results/3_carboplatin/), all batches/drugs and all models --
# case-specific filtering to cisplatin/carboplatin happens downstream, same
# as it already does for the raw 0_xevaset_extraction_summary outputs:
#   {Cohort}_batch_sensitivity_10_28day.csv:
#     batch.name, slope.control, slope.treatment, angle
#   {Cohort}_model_sensitivity_10_28day.csv:
#     model.id, mRECIST

library(Xeva)

RES_MEASURE <- c("slope", "angle", "mRECIST")
MIN_TIME <- 10
MAX_TIME <- 28

output_dir <- "../results/3_carboplatin"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

recompute_and_save <- function(rds_path, cohort_name) {
  x.set <- readRDS(rds_path)
  x.set <- setResponse(
    x.set, res.measure = RES_MEASURE,
    min.time = MIN_TIME, max.time = MAX_TIME, verbose = FALSE
  )

  batch_cols <- c("batch.name", "slope.control", "slope.treatment", "angle")
  batch_df <- x.set@sensitivity$batch[, batch_cols]
  batch_out <- file.path(output_dir, paste0(cohort_name, "_batch_sensitivity_10_28day.csv"))
  write.csv(batch_df, batch_out, row.names = FALSE)
  cat("Saved:", batch_out, "(", nrow(batch_df), "batches )\n")

  model_df <- x.set@sensitivity$model[, c("model.id", "mRECIST")]
  model_out <- file.path(output_dir, paste0(cohort_name, "_model_sensitivity_10_28day.csv"))
  write.csv(model_df, model_out, row.names = FALSE)
  cat("Saved:", model_out, "(", nrow(model_df), "models )\n")
}

recompute_and_save("../rawdata/mcgill_breast/Xeva_McGill.rds", "McGill")
recompute_and_save("../rawdata/uhn_breast/UHN_Cescon_Breast_DrugResponse_2025_v1.rds", "UHN")
