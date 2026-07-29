# Case Study 4 (Paclitaxel PDX Multi-omics Classification) — Draft Manuscript Text

*Drafted for Nature Communications style. Numbers verified against `1-data_preprocessing.ipynb`, `2-ML.ipynb`, and `3-figures.ipynb` as of 2026-07-28. Treat as a first draft — check tone/claims against the rest of the manuscript before finalizing.*

---

## Results

We assembled 80 paclitaxel-treated PDX batches from the UHN breast biobank and defined the primary response endpoint as a median split of batch-level growth-curve angle (Sensitive/Resistant, n = 40/40) rather than using majority-vote mRECIST directly; the two labels agreed in 76.2% of batches, with disagreement concentrated at the ambiguous stable-disease boundary, and since RNA-seq was profiled pre-treatment this choice avoids label-expression circularity.

Using fold-safe feature selection and four tuned classifiers evaluated over 25 repeated cross-validation folds, RNA expression (67 models, 49,348 genes) was the best single-omics predictor by a clear margin (random forest AUROC 0.732, 95% CI 0.684–0.780), exceeding CNV (62 models, AUROC 0.559, not significant after FDR) and mutation (62 models, AUROC 0.582). Combining CNV and mutation with RNA via early or late fusion did not improve on RNA alone (0.683 and 0.671 vs. 0.700 on the matched 62-model cohort, overlapping confidence intervals), consistent with the weaker omics diluting rather than complementing RNA's signal. Fifteen of 23 model/omics combinations tested were significant after Benjamini-Hochberg correction.

A random forest fit on the full RNA cohort for feature ranking (not performance estimation) showed known taxane-resistance genes (*TUBB*, *ABCB1*, *VIM*, *ZEB1*) retained with above-average importance without dominating the top features, indicating signal beyond previously catalogued genes. Pathway enrichment (GSEA against Hallmark/KEGG/GO Biological Process) showed Resistant models enriched for EMT and inflammatory/immune signaling — established resistance mechanisms — while Sensitive models were enriched for E2F targets and DNA repair, consistent with paclitaxel's mechanism as a mitotic-spindle poison. Given the modest cohort size and non-independence of repeated-CV folds, these findings, while FDR-corrected, should be read as hypothesis-generating rather than confirmatory.

---

## Methods

### PDX cohort and paclitaxel response phenotyping

PDX drug-sensitivity records were drawn from the extracted UHN breast cancer XevaSet, comprising batch-level (`batch_sensitivity.csv`), model-level (`model_sensitivity.csv`), and experimental-design (`expDesign.csv`) tables. Batches were filtered to single-agent paclitaxel treatment arms, excluding resistant-derivative sublines. Batch-level mRECIST categories were derived from the per-mouse mRECIST calls recorded in `model_sensitivity.csv` by majority vote across each batch's treatment-arm mice, with ties broken toward the worse response (PD > SD > PR > CR); this mirrors the aggregation method used for the carboplatin case study on the same PDX resource. The primary response label used for all downstream modeling was a binary median split of the batch-level growth-curve angle metric: the median was computed once across the full 80-batch cohort and applied as a fixed threshold to every sample subset used downstream (Sensitive: angle ≥ threshold; Resistant: angle < threshold), rather than being recomputed independently within each differently sized omics subset.

### Multi-omics data processing

**RNA-seq.** Raw expression values were log2(x + 1)-transformed, and genes with zero variance across all profiled samples or within the paclitaxel-treated subset were removed, yielding a matrix of 67 models × 49,348 genes.

**Copy number (CNV).** Absolute gene-level copy number was converted to a ploidy-relative continuous log2 ratio. For each sample, a baseline copy number was defined as the modal copy-number value across the full, unfiltered genome (i.e., computed before any gene-level filtering), floored at 1 to avoid division artifacts in samples with degenerate low baselines. Genes with missingness ≥ 20% across samples were removed, then each remaining value was transformed as log2((CN + 0.5) / baseline); remaining missing values were set to 0 (the neutral, at-baseline value on this scale). This yielded a matrix of 62 models × 10,313 genes. This approach mirrors the ploidy-relative baseline procedure used for CNV in the carboplatin case study on the same PDX resource, extended here to a continuous ratio rather than a binary amplification/deletion call to preserve graded copy-number information.

**Mutation.** Variant calls were binarized per gene per sample using an inclusive functional-variant whitelist (Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del, Frame_Shift_Ins, In_Frame_Del, In_Frame_Ins, Splice_Site, Translation_Start_Site, Nonstop_Mutation), consistent with the mutation-handling convention used for the carboplatin case study. This yielded a matrix of 62 models × 13,029 genes, covering an identical model set to the CNV data. The three-omics overlap (RNA ∩ CNV ∩ Mutation) comprised 62 models.

### Fold-safe feature selection

To avoid information leakage, feature selection was performed independently within each cross-validation training fold and applied unchanged to the corresponding held-out test fold. RNA features were selected by a variance prefilter (top 2,200 genes), a pairwise-correlation filter (Pearson |r| > 0.85, retaining the earlier-ranked gene of each correlated pair), and a final variance filter (top 2,000 genes). CNV features were selected by a positive-variance filter followed by a correlation filter (|r| > 0.95). Mutation features were selected by a prevalence filter (present in ≥ 10 and ≤ 90% of training samples). For all three omics layers, any gene from a curated 69-gene paclitaxel-prior list present in the training data was force-retained in addition to the filter-selected genes.

### Model training and evaluation

Four classifier families were evaluated: L1- and L2-penalized logistic regression (`liblinear` solver, class-balanced weighting), an RBF-kernel support vector machine (probability estimates enabled, class-balanced weighting), and a random forest (500 trees, `sqrt` max features, class-balanced weighting). Hyperparameters (regularization strength for logistic regression; C and gamma for SVM; max depth and minimum leaf size for random forest) were tuned by grid search with 3-fold stratified inner cross-validation, optimizing AUROC. Outer performance was assessed by 5-fold stratified cross-validation repeated 5 times (25 total folds; `random_state = 42` throughout).

Three modeling architectures were compared: (1) single-omics classification, with RNA evaluated both on the full 67-model RNA-profiled cohort and on the 62-model cohort paired across all three omics for direct comparability; (2) early fusion, concatenating each fold's selected RNA, CNV, and mutation features before classification on the 62-model paired cohort; and (3) late fusion / stacking, in which a base classifier was trained per omics layer via out-of-fold predictions on the training fold (inner 3-fold cross-validation) and combined by an L2-penalized logistic-regression meta-model, with a sensitivity analysis comparing logistic regression, random forest, and SVM-RBF as the base-model family.

### Statistical analysis

For each model/omics/analysis combination, we report the mean AUROC across the 25 repeated cross-validation folds, a 95% confidence interval based on the t-distribution, and a one-sample t-test against chance performance (AUROC = 0.5). Because repeated cross-validation folds resample the same underlying cohort and are not fully statistically independent, this test is the standard (if somewhat anti-conservative) approach used across most repeated-CV machine-learning reporting, rather than a variance-corrected alternative such as a Nadeau-Bengio test; p-values should be interpreted alongside the reported confidence intervals. Benjamini-Hochberg false discovery rate correction was applied across all 23 model/omics/analysis combinations tested.

### Feature importance and pathway enrichment analysis

To identify RNA features driving classification, we fit a single random forest (identical hyperparameters to the tuned single-omics model) on the full 67-model RNA cohort after applying the same fold-safe-style feature selection to the entire dataset; this model was used only to rank impurity-based feature importances and was not used for any performance estimate. For pathway-level analysis, all 49,348 profiled genes (not the pre-filtered classification feature subset) were ranked by a Welch's t-statistic contrasting Sensitive and Resistant models, and tested by pre-ranked gene set enrichment analysis (GSEA; `gseapy` v1.1.11) against the MSigDB Hallmark (2020), KEGG (2021 Human), and GO Biological Process (2023) gene set libraries (Enrichr collections), using 5,000 permutations, a minimum/maximum gene-set size of 15/500, and a fixed random seed (42). Gene sets were considered significant at a within-collection Benjamini-Hochberg FDR < 0.05.

---

*Files referenced: `4_paclitaxel/1-data_preprocessing.ipynb`, `4_paclitaxel/2-ML.ipynb`, `4_paclitaxel/3-figures.ipynb`, `4_paclitaxel/feature_sets.py`, `4_paclitaxel/pipeline_utils.py`. Result tables: `results/4_paclitaxel/classification_model_summary.csv`, `results/4_paclitaxel/rna_feature_importance.csv`, `results/4_paclitaxel/rna_gsea_*.csv`.*
