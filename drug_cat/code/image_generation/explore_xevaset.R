library(Xeva)

xset_path  <- "../Xeva_PDXE.rds"
output_dir <- "/Users/guanqiaofeng/Documents/BHK/XevaDB/case_study/drug_cat/data/rawdata/images"
n_figures  <- 1200

x_max <- 50
y_min <- -1
y_max <- 10

control_col   <- "#a6611a"
treatment_col <- "#018571"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

x.set <- readRDS(xset_path)

all_batches <- batchInfo(x.set)
cat("Total Batches:", length(all_batches), "\n")

has_control_and_treatment <- function(batch_id, x.set) {
  info <- batchInfo(x.set, batch = batch_id)[[batch_id]]
  has_control <- !is.null(info$control)   && length(info$control)   > 0 && info$control   != ""
  has_treat   <- !is.null(info$treatment) && length(info$treatment) > 0 && info$treatment != ""
  has_control && has_treat
}

batches_keep <- all_batches[
  vapply(all_batches, has_control_and_treatment, logical(1), x.set = x.set)
]
cat("Batches with control+treatment:", length(batches_keep), "\n")

stopifnot(length(batches_keep) >= n_figures)

set.seed(1)
batches_1200 <- sample(batches_keep, n_figures)

# manifest uses webp filenames
manifest <- data.frame(
  figure_id  = sprintf("fig_%06d", seq_along(batches_1200)),
  batch_id   = batches_1200,
  image_file = paste0(sprintf("fig_%06d", seq_along(batches_1200)), ".png"),
  stringsAsFactors = FALSE
)

failed <- character(0)

for (i in seq_along(batches_1200)) {
  b <- batches_1200[i]
  out_png <- file.path(output_dir, manifest$image_file[i])  # manifest should be .png here
  
  ok <- !inherits(
    try({
      p <- plotPDX(
        x.set,
        batch = b,
        vol.normal = TRUE,
        max.time = x_max,
        concurrent.time = TRUE,
        control.col = control_col,
        treatment.col = treatment_col,
        major.line.size = 1,
        title = b
      ) +
        coord_cartesian(
          xlim = c(0, x_max),
          ylim = c(y_min, y_max)
        ) +
        scale_x_continuous(expand = c(0, 0)) +
        theme_bw(base_size = 14) +
        theme(
          axis.title.x = element_text(size = 15),
          axis.title.y = element_text(size = 15),
          axis.text    = element_text(size = 13),
          
          legend.position = c(0.02, 0.98),
          legend.justification = c("left", "top"),
          legend.background = element_rect(
            fill = scales::alpha("white", 0.7),
            color = NA
          ),
          legend.text = element_text(size = 12),
          
          plot.title = element_text(size = 15, face = "bold")
        )
      
      ggsave(
        filename = out_png,
        plot = p,
        device = "png",  # force png
        width = 8, height = 6, units = "in", dpi = 120
      )
    }, silent = TRUE),
    "try-error"
  )
  
  if (!ok) {
    failed <- c(failed, b)
    if (file.exists(out_png)) file.remove(out_png)
  }
  
  if (i %% 100 == 0) cat("Saved", i, "of", length(batches_1200), "\n")
}

cat("Failed:", length(failed), "\n")

cat("PNG written:", length(list.files(output_dir, pattern="\\.png$", ignore.case=TRUE)), "\n")

write.csv(manifest, file.path(output_dir, "manifest.csv"), row.names = FALSE)
writeLines(if (length(failed) == 0) "NONE" else failed,
           file.path(output_dir, "failed_batches.txt"))

# Step 1: Convert PNG → WebP
pngs <- list.files(
  output_dir,
  pattern = "\\.png$",
  full.names = TRUE
)

length(pngs)   # should be ~1200

for (f in pngs) {
  out_webp <- sub("\\.png$", ".webp", f)
  system(sprintf(
    "cwebp -q 85 %s -o %s",
    shQuote(f),
    shQuote(out_webp)
  ))
}
# Step 2: Verify conversion
length(list.files(output_dir, pattern = "\\.webp$"))
file.info(list.files(output_dir, pattern = "\\.webp$", full.names = TRUE)[1:3])$size

# Step 3: Update your manifest
manifest$image_file <- sub("\\.png$", ".webp", manifest$image_file)
write.csv(manifest, file.path(output_dir, "manifest.csv"), row.names = FALSE)
