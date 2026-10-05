### McGill Breast XevaSet
# - read in xevaset of mcgill breast located in ../data/rawdata/mcgill_breast
# - extract experiment/expDesign/drug/models/omics/modToBiobaseMap
# - extract batch-level sensitivity table (slope, angle, AUC, TGI, mRECIST, ...)
# - extract model-level sensitivity table (mRECIST, best.response, ...)
# - output csv files under ../data/procdata/0_datasets/mcgill_breast

library(Xeva)
library(Biobase)

# ---- paths ----
raw_path   <- "../data/rawdata/mcgill_breast/Xeva_McGill.rds"
output_dir <- "../data/procdata/0_datasets/mcgill_breast"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# ---- read in xevaset ----
x.set <- readRDS(raw_path)

# ---- extract model table ----
model_df <- x.set@model
write.csv(model_df, file.path(output_dir, "models.csv"), row.names = FALSE)

# ---- extract drug table ----
drug_df <- x.set@drug
write.csv(drug_df, file.path(output_dir, "drug.csv"), row.names = FALSE)

# ---- extract experiment table (per-mouse growth curves) ----
experiment_list <- lapply(names(x.set@experiment), function(id) {
  e <- x.set@experiment[[id]]

  drug_name <- if (!is.null(e@drug$join.name)) {
    e@drug$join.name
  } else {
    paste(unlist(e@drug), collapse = "+")
  }

  df <- e@data
  df$model.id <- e@model.id
  df$drug.id  <- drug_name
  df
})

experiment_df <- do.call(rbind, experiment_list)
experiment_df <- experiment_df[, c("model.id", "drug.id", setdiff(colnames(experiment_df), c("model.id", "drug.id")))]

write.csv(experiment_df, file.path(output_dir, "experiment.csv"), row.names = FALSE)

# ---- extract experimental design table (control/treatment batch membership) ----
expDesign_list <- lapply(x.set@expDesign, function(batch) {
  data.frame(
    batch.name = batch$batch.name,
    model.id   = c(batch$control, batch$treatment),
    arm        = c(
      rep("control", length(batch$control)),
      rep("treatment", length(batch$treatment))
    ),
    stringsAsFactors = FALSE
  )
})

expDesign_df <- do.call(rbind, expDesign_list)
write.csv(expDesign_df, file.path(output_dir, "expDesign.csv"), row.names = FALSE)

# ---- extract model-to-omics mapping ----
map_df <- x.set@modToBiobaseMap
write.csv(map_df, file.path(output_dir, "modToBiobaseMap.csv"), row.names = FALSE)

# ---- extract omics (molecular profiles) matrices + gene annotation ----
for (mDataType in names(x.set@molecularProfiles)) {
  eset <- x.set@molecularProfiles[[mDataType]]

  mat <- exprs(eset)
  write.csv(mat, file.path(output_dir, paste0(mDataType, "_matrix.csv")))

  gene_anno <- fData(eset)
  write.csv(gene_anno, file.path(output_dir, paste0(mDataType, "_gene_annotation.csv")))
}

# ---- extract batch-level sensitivity table (McGill-specific: drives case study 3) ----
batch_sensitivity_df <- x.set@sensitivity$batch
write.csv(batch_sensitivity_df, file.path(output_dir, "batch_sensitivity.csv"), row.names = FALSE)

# ---- extract model-level sensitivity table (mRECIST, best.response, ...) ----
model_sensitivity_df <- x.set@sensitivity$model
write.csv(model_sensitivity_df, file.path(output_dir, "model_sensitivity.csv"), row.names = FALSE)

########
# end  #
########
