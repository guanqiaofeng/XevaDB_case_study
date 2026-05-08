# In R
library(genefu)
data(pam50)

# set current working directory to the script's location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load your data
exp_data <- read.csv("../data/procdata/1_doublingRate/rna_doubling_time_merged.csv", row_index=1)
# drop column "global_doubling_time"
exp_data <- exp_data[, !colnames(exp_data) %in% c("global_doubling_time")]

# pam50 mapping (ensure your column names are Gene Symbols)
# genefu expects genes as columns
annot <- data.frame("intercept"=rep(1, ncol(exp_data)), "row.names"=colnames(exp_data))
colnames(annot) <- "symbol"

# Run classification
pam50_res <- intrinsic.cluster.predict(sbt.model=pam50, data=exp_data, 
                                       annot=annot, do.mapping=TRUE)

# Export results
output <- data.frame(modelID = rownames(exp_data), subtype = pam50_res$subtype)
write.csv(output, "../data/procdata/1_doublingRate/pam50_predictions.csv", row.names=FALSE)