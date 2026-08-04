### Diagnose whether/how the case-4 response window should be chosen, using the
# same logic as case 1's MAX_DAY justification and case 3's window-choice
# diagnostic (window_confound_sweep.R): characterize the data's own confound
# structure (does the uncensored treatment-arm angle correlate with how long a
# model was followed, and how does the retained median angle drift as the
# window widens) rather than picking whichever window maximizes a downstream
# performance metric.
#
# Scope: the 80 paclitaxel batches used by case study 4 (batch.name contains
# "PACLITAXEL", excluding resistant-derivative lines containing "-"/"RES" --
# same filter as 1-data_preprocessing.ipynb). Same raw UHN XevaSet as case 3,
# but case 3's carboplatin batches/window choice are not reused here: it's a
# different drug with its own molecules/response dynamics.
#
# Same min.time=10 and max.time sweep range (14-56 by 7) as case 3's
# window_confound_sweep.R, so the two case studies use one consistent
# methodology: the deepest point (most extreme median response) in the sweep
# is the chosen cutoff, not a downstream-performance-maximizing choice.
#
# Output (data/procdata/4_paclitaxel/):
#   paclitaxel_angle_full_duration.csv: batch.name, slope.control, slope.treatment, angle (max.time=NULL)
#   paclitaxel_angle_sweep.csv: batch.name, slope.control, slope.treatment, angle, max_time (14..56 by 7)

library(Xeva)

MIN_TIME <- 10
CANDIDATE_MAX_TIME <- seq(14, 56, by = 7)

raw_path <- "../data/rawdata/uhn_breast/UHN_Breast_XevaSet_v2025.rds"
output_dir <- "../data/procdata/4_paclitaxel"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

x.set <- readRDS(raw_path)

all_batches <- batchInfo(x.set)
pax_batches <- all_batches[grepl("PACLITAXEL", all_batches) &
                              !grepl("-", all_batches) &
                              !grepl("RES", all_batches)]
cat("Paclitaxel batches:", length(pax_batches), "\n")

batch_cols <- c("batch.name", "slope.control", "slope.treatment", "angle")

full <- setResponse(x.set, res.measure = c("slope", "angle"), min.time = MIN_TIME, max.time = NULL, verbose = FALSE)
full_df <- full@sensitivity$batch[, batch_cols]
full_df <- full_df[full_df$batch.name %in% pax_batches, ]
write.csv(full_df, file.path(output_dir, "paclitaxel_angle_full_duration.csv"), row.names = FALSE)
cat("Full-duration:", nrow(full_df), "batches\n")

sweep_rows <- list()
for (mt in CANDIDATE_MAX_TIME) {
  res <- setResponse(x.set, res.measure = c("slope", "angle"), min.time = MIN_TIME, max.time = mt, verbose = FALSE)
  bdf <- res@sensitivity$batch[, batch_cols]
  bdf <- bdf[bdf$batch.name %in% pax_batches, ]
  bdf$max_time <- mt
  sweep_rows[[as.character(mt)]] <- bdf
}
sweep_df <- do.call(rbind, sweep_rows)
write.csv(sweep_df, file.path(output_dir, "paclitaxel_angle_sweep.csv"), row.names = FALSE)
cat("Sweep:", nrow(sweep_df), "rows across", length(CANDIDATE_MAX_TIME), "candidate max.time values\n")
