### Generic two-group limma moderated-t ranking statistic computation.
#
# Replaces the naive Welch's t-statistic previously used to rank genes for
# GSEA (scipy.stats.ttest_ind(..., equal_var=False)) with limma's empirical-
# Bayes moderated t-statistic: per-gene variance estimates are shrunk toward
# a prior fitted across the whole population of genes measured on the same
# samples, which is the more principled estimator when per-gene sample sizes
# are small (as small as 9 per group for the lung cohort here) -- the raw
# per-gene variance alone is a noisy estimate of that gene's true variance,
# and limma borrows strength across genes to stabilize it.
#
# Input: a samples x genes CSV (first column = sample id) with one additional
# grouping column (already computed by the caller, exactly two levels).
# Output: Gene, ranking_statistic (limma's moderated t for contrast_level
# minus reference_level -- sign convention matches whatever two-group
# ttest_ind(group_b, group_a) ordering the caller used previously).
#
# Usage:
#   Rscript compute_limma_ranking.R <input_csv> <group_column> <reference_level> <contrast_level> <output_csv>

suppressMessages(library(limma))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: Rscript compute_limma_ranking.R <input_csv> <group_column> <reference_level> <contrast_level> <output_csv>")
}
input_csv <- args[1]
group_column <- args[2]
reference_level <- args[3]
contrast_level <- args[4]
output_csv <- args[5]

data <- read.csv(input_csv, row.names = 1, check.names = FALSE)
group <- factor(data[[group_column]], levels = c(reference_level, contrast_level))
cat("Group counts:\n"); print(table(group))

expr <- data[, colnames(data) != group_column]
expr <- t(as.matrix(expr))  # genes x samples, as limma expects
storage.mode(expr) <- "numeric"

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

fit <- lmFit(expr, design)
contrast_formula <- paste(contrast_level, "-", reference_level)
contrast_matrix <- makeContrasts(contrasts = contrast_formula, levels = design)
fit2 <- eBayes(contrasts.fit(fit, contrast_matrix))

res <- topTable(fit2, number = Inf, sort.by = "none")
out <- data.frame(Gene = rownames(res), ranking_statistic = res$t, unadjusted_p_value = res$P.Value)
write.csv(out, output_csv, row.names = FALSE)
cat("Saved limma moderated-t ranking:", nrow(out), "genes to", output_csv, "\n")
