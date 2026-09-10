### Intrinsic molecular subtype (PAM50-equivalent) for the UHN breast PDX
# paclitaxel cohort, from RNA-seq alone -- there is no clinical ER/PR/HER2
# IHC annotation anywhere in the UHN breast XevaSet (models.csv, the
# modelInfo() slot, and 0_xevaset_extraction/ were all checked; none carry
# receptor status), so subtype has to come from expression.
#
# Uses AIMS (Absolute Intrinsic Molecular Subtyping, Paquet & Hallett 2015,
# JNCI) rather than genefu's centroid-based PAM50: genefu isn't installed,
# and more importantly AIMS is a single-sample, rank-based k-TSP classifier
# that doesn't require normalizing against a reference cohort the way
# centroid PAM50 does -- a better fit for a PDX RNA-seq set with no matched
# normal/reference panel. It reports the same five PAM50 labels (Basal,
# Her2, LumA, LumB, Normal).
#
# Input is the fully-processed rna_logtpm.csv (log2(TPM+1), the paclitaxel-
# cohort feature matrix 1-data_preprocessing.ipynb builds and later patches
# with the windowed response columns) -- gene columns are untouched by that
# later patch, so this can run any time after the RNA-seq section, and is
# unaffected by the min.time/max.time response-window choice.
#
# Output (results/4_paclitaxel/): uhn_breast_pam50_aims_subtype.csv
#   model.id, subtype, confidence (winning-class posterior probability),
#   prob_Basal, prob_Her2, prob_LumA, prob_LumB, prob_Normal

suppressMessages({
  library(AIMS)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

rna_path <- "../data/procdata/4_paclitaxel/rna_logtpm.csv"
output_dir <- "../results/4_paclitaxel"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
output_path <- file.path(output_dir, "uhn_breast_pam50_aims_subtype.csv")

rna <- read.csv(rna_path, row.names = 1, check.names = FALSE)
response_cols <- c("angle", "slope.treatment", "mRECIST")
rna <- rna[, !(colnames(rna) %in% response_cols)]
cat("RNA feature matrix:", nrow(rna), "models x", ncol(rna), "genes\n")

# Gene symbol -> Entrez ID (AIMS's gene pairs are keyed on Entrez). Symbols
# with no mapping, or a duplicate SYMBOL->ENTREZID row, are dropped -- AIMS
# only needs the subset of ~100 genes its k-TSP rules use, so losing
# unmapped/ambiguous symbols from the ~49k-gene matrix has no effect on it.
map <- AnnotationDbi::select(org.Hs.eg.db, keys = colnames(rna),
                              keytype = "SYMBOL", columns = "ENTREZID")
map <- map[!duplicated(map$SYMBOL) & !is.na(map$ENTREZID), ]
cat("Gene symbols mapped to Entrez ID:", nrow(map), "of", ncol(rna), "\n")

expr <- t(rna)  # genes x samples, as AIMS expects
common <- intersect(rownames(expr), map$SYMBOL)
expr <- expr[common, , drop = FALSE]
map <- map[match(common, map$SYMBOL), ]
rownames(expr) <- map$ENTREZID
storage.mode(expr) <- "numeric"

res <- applyAIMS(expr, rownames(expr))

cl <- res$cl[, 1]
probmat <- res$all.probs[["20"]]
rownames(probmat) <- names(cl)
confidence <- sapply(seq_along(cl), function(i) probmat[i, cl[i]])

out <- data.frame(
  model.id = names(cl),
  subtype = as.character(cl),
  confidence = round(confidence, 4),
  prob_Basal = round(probmat[, "Basal"], 4),
  prob_Her2 = round(probmat[, "Her2"], 4),
  prob_LumA = round(probmat[, "LumA"], 4),
  prob_LumB = round(probmat[, "LumB"], 4),
  prob_Normal = round(probmat[, "Normal"], 4),
  row.names = NULL
)
write.csv(out, output_path, row.names = FALSE)

cat("\nSubtype counts:\n")
print(table(out$subtype))
cat("\nSaved:", output_path, "(", nrow(out), "models )\n")
