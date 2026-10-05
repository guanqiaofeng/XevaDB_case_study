### Diagnose whether/how the case-3 response window (min.time=10, max.time=30)
# should be chosen, using the same logic as case 1's MAX_DAY justification:
# characterize the data's own confound structure (does the uncensored
# treatment-arm slope correlate with how long a model was followed, and how
# does the retained median slope drift as the window widens) rather than
# picking whichever window maximizes a downstream performance metric.
#
# Output (results/3_carboplatin/):
#   {Cohort}_slope_full_duration.csv: batch.name, slope.treatment (max.time=NULL)
#   {Cohort}_slope_sweep.csv: batch.name, slope.treatment, max_time (14..56 by 7)

library(Xeva)

MIN_TIME <- 10
CANDIDATE_MAX_TIME <- seq(14, 56, by = 7)

output_dir <- "../results/3_carboplatin"

run_cohort <- function(rds_path, cohort_name, batch_names) {
  x.set <- readRDS(rds_path)

  full <- setResponse(x.set, res.measure = c("slope", "angle"), min.time = MIN_TIME, max.time = NULL, verbose = FALSE)
  full_df <- full@sensitivity$batch[, c("batch.name", "slope.treatment")]
  full_df <- full_df[full_df$batch.name %in% batch_names, ]
  write.csv(full_df, file.path(output_dir, paste0(cohort_name, "_slope_full_duration.csv")), row.names = FALSE)
  cat(cohort_name, "full-duration:", nrow(full_df), "batches\n")

  sweep_rows <- list()
  for (mt in CANDIDATE_MAX_TIME) {
    res <- setResponse(x.set, res.measure = c("slope", "angle"), min.time = MIN_TIME, max.time = mt, verbose = FALSE)
    bdf <- res@sensitivity$batch[, c("batch.name", "slope.treatment")]
    bdf <- bdf[bdf$batch.name %in% batch_names, ]
    bdf$max_time <- mt
    sweep_rows[[as.character(mt)]] <- bdf
  }
  sweep_df <- do.call(rbind, sweep_rows)
  write.csv(sweep_df, file.path(output_dir, paste0(cohort_name, "_slope_sweep.csv")), row.names = FALSE)
  cat(cohort_name, "sweep:", nrow(sweep_df), "rows across", length(CANDIDATE_MAX_TIME), "candidate max.time values\n")
}

mcgill_batches <- read.delim(file.path(output_dir, "McGill_cisplatin_models.tsv"))$batch.name
uhn_batches <- read.delim(file.path(output_dir, "UHN_carboplatin_models.tsv"))$batch.name

run_cohort("../rawdata/mcgill_breast/Xeva_McGill.rds", "McGill", mcgill_batches)
run_cohort("../rawdata/uhn_breast/UHN_Breast_XevaSet_v2025.rds", "UHN", uhn_batches)
