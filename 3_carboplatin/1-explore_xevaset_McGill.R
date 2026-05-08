library(Xeva)

# Define your desired directory
output_dir <- "/Users/guanqiaofeng/Documents/BHK/XevaDB/case_study/carboplatin/data/procdata/McGill"

# Check if the directory exists; if not, create it
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Set working directory
setwd(output_dir)

# read in xevaset
x.set <- readRDS("../data/rawdata/mcgill_breast/Xeva_McGill.rds")

# extract batch response table
resp_df <- x.set@sensitivity$batch
write.csv(resp_df, "../data/procdata/3_carboplatin/McGill_batch_response_table.csv", row.names = FALSE)

# extract mutation table
names(x.set@molecularProfiles)
mut <- x.set@molecularProfiles$mutation
mut_mat <- exprs(mut)
write.csv(mut_mat, "../data/procdata/3_carboplatin/McGill_mutation_matrix.csv")

# extract cnv table
cnv <- x.set@molecularProfiles$CNV
cnv_mat <- exprs(cnv)
write.csv(cnv_mat, "../data/procdata/3_carboplatin/McGill_cnv_matrix.csv")

# extract model-to-omics mapping
map_df <- x.set@modToBiobaseMap

write.csv(map_df,
          "../data/procdata/3_carboplatin/McGill_modToBiobaseMap.csv",
          row.names = FALSE)


########
# end #
########
