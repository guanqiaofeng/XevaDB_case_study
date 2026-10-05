### Recompute drug-sensitivity metrics for the UHN breast XevaSet's paclitaxel
# batches using an explicit 10-49 day response window (min.time = 10,
# max.time = 49), replacing the as-extracted values in
# 0_datasets/uhn_breast/batch_sensitivity.csv, which reflect Xeva's default/
# full observed treatment duration (median follow-up 135 days, up to 538) --
# the same "full-duration, undocumented window" issue found in case 3's UHN
# carboplatin arm (see 3_carboplatin/recompute_sensitivity_10_28day.R), here
# for the paclitaxel batches used by case study 4: slope + angle at the batch
# level, mRECIST at the model (per-mouse) level (batch-level mRECIST is then
# derived downstream in 1-data_preprocessing.ipynb by majority vote across
# each batch's treatment-arm mice, ties broken toward the more resistant
# call -- same as before, just fed freshly-windowed mRECIST calls).
#
# min.time=10 is inherited from case 3's justification, not re-derived here:
# it is a structural property of Xeva's mRECIST()/.getBestResponse() (day 0
# is a trivial, always-available "best response" candidate at min.time=0,
# which makes the PD category unreachable), not something specific to the
# drug or cohort. It has ZERO effect on slope()/angle().
#
# Window choice (max.time=49): see window_confound_sweep.R / the
# "Window-choice justification" cells in 1-data_preprocessing.ipynb. Same
# min.time=10, max.time sweep range (14-56 by 7) and same deepest-point
# selection criterion as case 3: rather than picking whichever window
# maximizes downstream model performance, the sweep's median response
# reaches its most extreme (deepest, least regrowth-contaminated) value at
# max.time=49 for BOTH candidate response quantities -- angle peaks at
# 112.4 degrees and slope.treatment troughs at -41.5 degrees, both at the
# same 49-day point -- then erodes back toward non-responding at wider
# windows, mirroring case 3's McGill result exactly (deepest point at 28
# days there). 49 is the data's own inflection point, not a value chosen to
# maximize downstream classification performance.
#
# Output (results/4_paclitaxel/), paclitaxel batches/models only (the
# 80 batches selected by the same PACLITAXEL/non-"-"/non-RES filter used in
# 1-data_preprocessing.ipynb):
#   paclitaxel_batch_sensitivity_min10_max49.csv:
#     batch.name, slope.control, slope.treatment, angle
#   paclitaxel_model_sensitivity_min10_max49.csv:
#     model.id, mRECIST

library(Xeva)

RES_MEASURE <- c("slope", "angle", "mRECIST")
MIN_TIME <- 10
MAX_TIME <- 49

raw_path <- "../rawdata/uhn_breast/UHN_Breast_XevaSet_v2025.rds"
output_dir <- "../results/4_paclitaxel"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

x.set <- readRDS(raw_path)

all_batches <- batchInfo(x.set)
pax_batches <- all_batches[grepl("PACLITAXEL", all_batches) &
                              !grepl("-", all_batches) &
                              !grepl("RES", all_batches)]

x.set <- setResponse(
  x.set, res.measure = RES_MEASURE,
  min.time = MIN_TIME, max.time = MAX_TIME, verbose = FALSE
)

batch_cols <- c("batch.name", "slope.control", "slope.treatment", "angle")
batch_df <- x.set@sensitivity$batch[, batch_cols]
batch_df <- batch_df[batch_df$batch.name %in% pax_batches, ]
batch_out <- file.path(output_dir, "paclitaxel_batch_sensitivity_min10_max49.csv")
write.csv(batch_df, batch_out, row.names = FALSE)
cat("Saved:", batch_out, "(", nrow(batch_df), "batches )\n")

# Model-level mRECIST covers every mouse (control + treatment, all drugs) --
# restricting here to the treatment-arm mice belonging to paclitaxel batches,
# matching what 1-data_preprocessing.ipynb's majority-vote step consumes.
expDesign_list <- lapply(x.set@expDesign, function(b) {
  data.frame(batch.name = b$batch.name, model.id = c(b$control, b$treatment),
             arm = c(rep("control", length(b$control)), rep("treatment", length(b$treatment))),
             stringsAsFactors = FALSE)
})
expDesign_df <- do.call(rbind, expDesign_list)
pax_treatment_mice <- unique(expDesign_df$model.id[expDesign_df$batch.name %in% pax_batches &
                                                       expDesign_df$arm == "treatment"])

model_df <- x.set@sensitivity$model[, c("model.id", "mRECIST")]
model_df <- model_df[model_df$model.id %in% pax_treatment_mice, ]
model_out <- file.path(output_dir, "paclitaxel_model_sensitivity_min10_max49.csv")
write.csv(model_df, model_out, row.names = FALSE)
cat("Saved:", model_out, "(", nrow(model_df), "models )\n")
