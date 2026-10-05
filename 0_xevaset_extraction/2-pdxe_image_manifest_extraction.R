### PDXE pan-cancer XevaSet -- image manifest for case study 4
# - read in xevaset of PDXE located in ../rawdata/pdxe
# - restrict to control+treatment batches with a non-missing model-level mRECIST call
# - stratified sample of n_figures batches: all eligible CR/PR retained, SD capped
#   at half the remaining budget, PD filling the remainder (seed = 1; see Methods).
#   Sampling uses the RDS's stored, full-duration mRECIST.
# - recompute model- and batch-level sensitivity metrics for the sampled batches
#   using a min.time = 10, max.time = 50 day window to match the rendered images
#   (the stored sensitivity slot instead covers the full treatment duration).
# - TGI is dropped: Xeva's TGI() errors on any batch with no data points in
#   [10, 50] on one arm, which occurs for several batches in the full PDXE set.
# - render one tumor-volume plot per sampled batch and convert PNG -> WebP
# - output images and manifest.csv under ../procdata/pdxe
# - Requires the cwebp CLI (https://developers.google.com/speed/webp/docs/cwebp)
#   on PATH for the PNG -> WebP conversion step.

library(Xeva)
library(dplyr)
library(ggplot2)

# ---- paths ----
xset_path  <- "../rawdata/pdxe/Xeva_PDXE.rds"
output_dir <- "../procdata/pdxe"
n_figures  <- 1300

# Plotting limits
x_max <- 50
y_min <- -1
y_max <- 10

# Color palette: earthy brown for control, deep teal for treatment
control_col   <- "#a6611a"
treatment_col <- "#018571"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ---- read in xevaset ----
x.set <- readRDS(xset_path)
model_metrics <- methods::slot(x.set, "sensitivity")$model
cat("Total raw models in sensitivity slot:", nrow(model_metrics), "\n")

# ---- filter for control+treatment pairs ----
exp_design_df <- data.frame(
  batch.id  = names(methods::slot(x.set, "expDesign")),
  control   = sapply(methods::slot(x.set, "expDesign"), function(x) paste(x$control, collapse = ",")),
  treatment = sapply(methods::slot(x.set, "expDesign"), function(x) paste(x$treatment, collapse = ",")),
  stringsAsFactors = FALSE
)

valid_batches <- exp_design_df %>%
  filter(control != "" & !is.na(control))

cat("Total batches in expDesign:", nrow(exp_design_df), "\n")
cat("Batches passing control+treatment filter:", nrow(valid_batches), "\n")

# ---- intersect metrics with valid batches, requiring a non-missing mRECIST call ----
valid_pool <- model_metrics %>%
  filter(model.id %in% valid_batches$treatment, !is.na(mRECIST))

cat("Models with both control+treatment data and a valid mRECIST label:", nrow(valid_pool), "\n")

# ---- stratified sampling: all CR/PR retained, SD/PD split the remainder ----
final_CR <- valid_pool %>% filter(mRECIST == "CR")
final_PR <- valid_pool %>% filter(mRECIST == "PR")
final_SD_pool <- valid_pool %>% filter(mRECIST == "SD")
final_PD_pool <- valid_pool %>% filter(mRECIST == "PD")

n_responders <- nrow(final_CR) + nrow(final_PR)
n_remaining  <- n_figures - n_responders

cat("Responders - CR:", nrow(final_CR), "| PR:", nrow(final_PR), "\n")
cat("Targeting", n_remaining, "more images from SD/PD pools...\n")

set.seed(1)
n_sd_to_take <- min(nrow(final_SD_pool), floor(n_remaining / 2))
sampled_SD   <- final_SD_pool[sample(nrow(final_SD_pool), n_sd_to_take), ]

n_pd_to_take <- n_figures - (n_responders + nrow(sampled_SD))
sampled_PD   <- final_PD_pool[sample(nrow(final_PD_pool), n_pd_to_take), ]

balanced_metrics <- rbind(final_CR, final_PR, sampled_SD, sampled_PD)
balanced_ids <- merge(
  balanced_metrics[, "model.id", drop = FALSE],
  valid_batches[, c("batch.id", "treatment")],
  by.x = "model.id", by.y = "treatment"
)

cat("Final sampling successful. Total images to generate:",
    nrow(balanced_ids), "\n")

# ---- recompute model- and batch-level sensitivity metrics for the sampled
# batches, using the same 10-50 day response window as the rendered images ----
x.set.windowed <- setResponse(
  x.set, res.measure = c("mRECIST", "slope", "AUC", "angle", "abc"),
  min.time = 10, max.time = x_max, verbose = FALSE
)
model_metrics_windowed <- methods::slot(x.set.windowed, "sensitivity")$model
batch_metrics_windowed <- methods::slot(x.set.windowed, "sensitivity")$batch
batch_metrics_windowed_clean <- batch_metrics_windowed[, c(
  "batch.name", "slope.control", "slope.treatment", "angle",
  "auc.control", "auc.treatment", "abc"
)]

balanced_df_rich <- merge(
  balanced_ids, model_metrics_windowed, by = "model.id", all.x = TRUE
)
balanced_df_rich <- merge(
  balanced_df_rich, batch_metrics_windowed_clean,
  by.x = "batch.id", by.y = "batch.name", all.x = TRUE
)

# ---- build the figure/batch manifest ----
manifest <- data.frame(
  figure_id = sprintf("fig_%06d", seq_along(balanced_df_rich$batch.id)),
  batch_id  = balanced_df_rich$batch.id,
  model_id  = balanced_df_rich$model.id,

  model_mRECIST                    = balanced_df_rich$mRECIST,
  model_best.response               = balanced_df_rich$best.response,
  model_best.response.time          = balanced_df_rich$best.response.time,
  model_best.average.response       = balanced_df_rich$best.average.response,
  model_best.average.response.time  = balanced_df_rich$best.average.response.time,
  model_AUC                         = balanced_df_rich$AUC,
  model_slope                       = balanced_df_rich$slope,

  batch_slope.control   = balanced_df_rich$slope.control,
  batch_slope.treatment = balanced_df_rich$slope.treatment,
  batch_angle           = balanced_df_rich$angle,
  batch_auc.control     = balanced_df_rich$auc.control,
  batch_auc.treatment   = balanced_df_rich$auc.treatment,
  batch_abc             = balanced_df_rich$abc,

  image_file = paste0(sprintf("fig_%06d", seq_along(balanced_df_rich$batch.id)), ".png"),
  stringsAsFactors = FALSE
)

cat("mRECIST distribution in manifest:\n")
print(table(manifest$model_mRECIST))

# ---- render one tumor-volume plot per sampled batch (skips any batch whose
# webp already exists) ----
failed <- character(0)
n_skipped <- 0

for (i in seq_len(nrow(manifest))) {
  b <- manifest$batch_id[i]
  out_png <- file.path(output_dir, manifest$image_file[i])
  out_webp_existing <- sub("\\.png$", ".webp", out_png)
  if (file.exists(out_webp_existing)) {
    n_skipped <- n_skipped + 1
    next
  }
  plot_buffer <- x_max + 5
  x_min_pad <- -5
  y_min_pad <- -2

  ok <- !inherits(try({
    p <- plotPDX(
      x.set, batch = b, vol.normal = TRUE, max.time = x_max,
      concurrent.time = TRUE, control.col = control_col, treatment.col = treatment_col,
      major.line.size = 1.2, title = b
    ) +
      # Threshold line: tumor disappearance at y = -1, mapped to aes for legend inclusion
      geom_hline(aes(yintercept = -1, linetype = "tumour disappeared"),
                 color = "red", linewidth = 0.8) +
      scale_linetype_manual(name = NULL, values = c("tumour disappeared" = "dashed")) +
      guides(
        color = guide_legend(order = 1),
        linetype = guide_legend(order = 2)
      ) +
      coord_cartesian(xlim = c(x_min_pad, plot_buffer), ylim = c(y_min_pad, y_max)) +
      scale_x_continuous(expand = c(0, 0), breaks = seq(0, x_max, by = 10)) +
      scale_y_continuous(expand = c(0, 0), breaks = seq(-2, y_max, by = 2)) +
      theme_bw(base_size = 14) +
      theme(
        aspect.ratio = 0.8,
        legend.position = "top",
        legend.direction = "horizontal",
        legend.box = "horizontal",
        legend.title = element_blank(),
        legend.key = element_blank(),
        legend.background = element_blank(),
        legend.text = element_text(size = 15),
        axis.title = element_text(size = 15, face = "bold"),
        axis.text  = element_text(size = 15, color = "black"),
        axis.ticks = element_blank(),
        plot.title = element_text(size = 15, face = "bold", hjust = 0.5),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = "black"),
        panel.grid.major = element_line(color = "#f0f0f0"),
        panel.grid.minor = element_blank()
      )

    ggsave(filename = out_png, plot = p, device = "png",
           width = 6.2, height = 6, units = "in", dpi = 120)
  }, silent = TRUE), "try-error")

  if (!ok) {
    failed <- c(failed, b)
    if (file.exists(out_png)) file.remove(out_png)
  }

  if (i %% 100 == 0) cat("Processed", i, "of", nrow(manifest), "\n")
}

cat("Skipped (webp already existed):", n_skipped, "\n")
cat("Failed:", length(failed), "\n")
cat("PNG written:", length(list.files(output_dir, pattern = "\\.png$", ignore.case = TRUE)), "\n")

write.csv(manifest, file.path(output_dir, "manifest.csv"), row.names = FALSE)
writeLines(if (length(failed) == 0) "NONE" else failed,
           file.path(output_dir, "failed_batches.txt"))

# ---- convert PNG -> WebP, update manifest, remove PNGs ----
pngs <- list.files(output_dir, pattern = "\\.png$", full.names = TRUE)
cat("PNGs to convert:", length(pngs), "\n")

for (f in pngs) {
  out_webp <- sub("\\.png$", ".webp", f)
  status <- system(sprintf("cwebp -q 85 %s -o %s", shQuote(f), shQuote(out_webp)))
  if (status == 0 && file.exists(out_webp)) {
    file.remove(f)
  }
}

cat("WebP files written:", length(list.files(output_dir, pattern = "\\.webp$")), "\n")

manifest$image_file <- sub("\\.png$", ".webp", manifest$image_file)
write.csv(manifest, file.path(output_dir, "manifest.csv"), row.names = FALSE)
