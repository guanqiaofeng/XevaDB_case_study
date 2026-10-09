# 2_drugSensitivity/ — Case study 2: image-based drug-response classification

Compares an image-based ResNet18 classifier against a feature-based Random
Forest on the same task: sorting PDXE tumour-volume growth curves into 5
ordinal drug-response categories, against a 3-expert consensus ground
truth. Maps to **Figure 3** and **Supplemental Figure 2**.

- **Environment**: two separate ones
  
    - `../requirements.txt` (RandomForest + tabular notebooks), and
    - `../requirements-resnet.txt` (`3-ml_resnet18.ipynb`, needs `torch`/`torchvision`; keep in its own kernel, do not merge).
- **Inputs**: `../rawdata/expert_rating`, `../procdata/pdxe/`.
- **Outputs**: `../results/2_drugSensitivity` and `../figures_tables/`.

## Run order

| Notebook | Purpose | Key outputs |
|---|---|---|
| `1-expertRating_preprocessing.ipynb` | Merge 3 experts' ratings (`rawdata/expert_rating/`) by majority vote; assign the 5-fold CV splits shared by every model trained below | `results/2_drugSensitivity/expert_rating/` consensus labels + fold assignments |
| `2-image_preprocessing.ipynb` | Crop the fixed title/metadata band off all 1,300 raw PDXE batch images (`procdata/pdxe/images/*.webp`) | `results/2_drugSensitivity/images_clean/*.png` |
| `3-ml_resnet18.ipynb` | Train/evaluate ResNet18 on the cleaned images, 5-fold CV | `results/2_drugSensitivity/ml_model/resnet18/` (fold checkpoints + predictions) |
| `4-ml_randomforest.ipynb` | Engineer 13 response features from `procdata/pdxe/images/manifest.csv`, train/evaluate Random Forest, same folds; SHAP on a final full-data fit | `results/2_drugSensitivity/ml_model/random_forest/` |
| `5-ml_model_interpretation.ipynb` | Pool both models' out-of-fold predictions; performance comparison, confusion matrices, Grad-CAM | Fig 3b–d, Supp Fig 2c–e, **Table 3**, **Table 4** |

`1` and `2` are independent of each other; both must finish before `3`/`4`;
`5` needs both `3` and `4`.
