import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import pearsonr, spearmanr

# =========================
# CONFIG & HELPERS (Matching ModelZoo)
# =========================
N_SPLITS, N_REPEATS = 5, 5
RANDOM_STATE = 42
N_HVG = 100
print(f"HVG Count for ML: {N_HVG}")
RF_PARAMS = dict(n_estimators=500, max_depth=6, min_samples_leaf=1, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)

EN_L1_RATIOS = [0.1, 0.5, 0.9]
EN_CS = [0.01, 0.1, 1.0, 10.0]
LR_CS = [0.01, 0.1, 1.0, 10.0]

def select_hvg_train_only(X_train_df, n_hvg):
    return X_train_df.var(axis=0).sort_values(ascending=False).head(n_hvg).index.tolist()

def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    r = spearmanr(a, b).correlation
    return float(0.0 if np.isnan(r) else r)

def safe_pearson(a, b):
    r = pearsonr(a, b)[0]
    return float(0.0 if np.isnan(r) else r)

# =========================
# MAIN ENGINE
# =========================
df = pd.read_csv("../data/procdata/1_doublingRate/rna_HVG2000_doubling_time_merged.csv", index_col=0)
y_raw = np.log1p(df["global_doubling_time"].astype(float)).values
X_df_full = df.drop(columns=["global_doubling_time"])
model_ids = df.index.values

rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
metrics = {m: {"AUROC": [], "Spearman": [], "Pearson": []} for m in ["ElasticNet", "RandomForest", "LassoedRF"]}

all_rf_importances = []
all_pred_records = []
# best_cs_lr = []
# best_cs_en = []
# best_l1_ratios = []
print("Running 25-fold CV with Inner-Tuning (ModelZoo Style)...")

for fold_idx, (tr_idx, te_idx) in enumerate(rkf.split(X_df_full)):
    # 1. Split Raw Data
    X_tr_df, X_te_df = X_df_full.iloc[tr_idx], X_df_full.iloc[te_idx]
    y_tr_raw, y_te_raw = y_raw[tr_idx], y_raw[te_idx]
    
    # 2. Thresholding on Training Set (Leakage-Safe)
    thr_low, thr_high = np.quantile(y_tr_raw, 0.33), np.quantile(y_tr_raw, 0.67)
    
    tr_mask = (y_tr_raw <= thr_low) | (y_tr_raw >= thr_high)
    te_mask = (y_te_raw <= thr_low) | (y_te_raw >= thr_high)
    
    X_tr_ext, y_tr_ext = X_tr_df.loc[tr_mask], (y_tr_raw[tr_mask] >= thr_high).astype(int)
    X_te_ext, y_te_ext = X_te_df.loc[te_mask], (y_te_raw[te_mask] >= thr_high).astype(int)
    
    # FIX: Correctly slice the continuous values for Spearman correlation
    y_te_raw_ext = y_te_raw[te_mask] 
    te_ids_ext = model_ids[te_idx][te_mask]
    
    if len(np.unique(y_te_ext)) < 2: 
        continue

    # 3. HVG Selection (Training Only)
    hvg = select_hvg_train_only(X_tr_ext, N_HVG)
    X_tr, X_te = X_tr_ext[hvg].values, X_te_ext[hvg].values

    # --- MODEL 1: RANDOM FOREST ---
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_tr, y_tr_ext)
    p_rf = rf.predict_proba(X_te)[:, 1]
    all_rf_importances.append(pd.Series(rf.feature_importances_, index=hvg))
    metrics["RandomForest"]["AUROC"].append(roc_auc_score(y_te_ext, p_rf))
    metrics["RandomForest"]["Spearman"].append(safe_spearman(p_rf, y_te_raw_ext))
    metrics["RandomForest"]["Pearson"].append(safe_pearson(p_rf, y_te_raw_ext))

    # --- MODEL 2: ELASTIC NET ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    en = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=EN_L1_RATIOS, Cs=EN_CS, cv=4,
                            max_iter=10000, n_jobs=-1, random_state=RANDOM_STATE, scoring='roc_auc').fit(X_tr_s, y_tr_ext)
    p_en = en.predict_proba(X_te_s)[:, 1]
    metrics["ElasticNet"]["AUROC"].append(roc_auc_score(y_te_ext, p_en))
    metrics["ElasticNet"]["Spearman"].append(safe_spearman(p_en, y_te_raw_ext))
    metrics["ElasticNet"]["Pearson"].append(safe_pearson(p_en, y_te_raw_ext))
    # current_best_c_en = en.C_[0]
    # best_cs_en.append(current_best_c_en)
    # print(f"EN Fold {fold_idx+1}: Best C = {current_best_c_en}")
    # # also store and print best l1_ratio
    # current_best_l1 = en.l1_ratio_[0]
    # best_l1_ratios.append(current_best_l1)
    # print(f"EN Fold {fold_idx+1}: Best l1_ratio = {current_best_l1}")

    # --- MODEL 3: LASSOED RF ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    Z_tr = ohe.fit_transform(rf.apply(X_tr))
    Z_te = ohe.transform(rf.apply(X_te))
    lrf = LogisticRegressionCV(penalty="l1", solver="saga", Cs=LR_CS, cv=4, max_iter=10000, 
                             n_jobs=-1, random_state=RANDOM_STATE, scoring='roc_auc').fit(Z_tr, y_tr_ext)
    p_lrf = lrf.predict_proba(Z_te)[:, 1]
    metrics["LassoedRF"]["AUROC"].append(roc_auc_score(y_te_ext, p_lrf))
    metrics["LassoedRF"]["Spearman"].append(safe_spearman(p_lrf, y_te_raw_ext))
    metrics["LassoedRF"]["Pearson"].append(safe_pearson(p_lrf, y_te_raw_ext))
    # current_best_c_lr = lrf.C_[0]
    # best_cs_lr.append(current_best_c_lr)
    # print(f"LR Fold {fold_idx+1}: Best C = {current_best_c_lr}")

    # 4. Log Everything
    for m, p in zip(["RandomForest", "ElasticNet", "LassoedRF"], [p_rf, p_en, p_lrf]):
        for i in range(len(te_ids_ext)):
            all_pred_records.append({
                "Fold": fold_idx + 1,
                "Model_Type": m,
                "ModelID": te_ids_ext[i],
                "True_Label": y_te_ext[i],
                "Prob_Slow": p[i],
            })
    
    if (fold_idx + 1) % 10 == 0: print(f"  Fold {fold_idx+1}/25 complete...")

# Output Results
print("\n===== FINAL RAW GROWTH CV RESULTS =====")
for name in ["ElasticNet", "LassoedRF", "RandomForest"]:
    aucs, sps, prs = metrics[name]["AUROC"], metrics[name]["Spearman"], metrics[name]["Pearson"]
    print(f"{name:12s} AUROC {np.mean(aucs):.3f} ± {np.std(aucs):.3f} | Spearman {np.mean(sps):.3f} ± {np.std(sps):.3f} | Pearson {np.mean(prs):.3f} ± {np.std(prs):.3f}")

# Create a DataFrame for plotting later
all_results = []
for model_name in metrics:
    for i in range(len(metrics[model_name]["AUROC"])):
        all_results.append({
            "Model": model_name,
            "Fold": i + 1,
            "AUROC": metrics[model_name]["AUROC"][i],
            "Spearman": metrics[model_name]["Spearman"][i],
            "Pearson": metrics[model_name]["Pearson"][i]
        })
    
results_df = pd.DataFrame(all_results)
results_df.to_csv("../data/procdata/1_doublingRate/ModelZoo_Fold_Results.csv", index=False)
print("Saved fold-level results to ModelZoo_Fold_Results.csv")

# =========================
# SAVE OUTPUTS
# =========================
pd.DataFrame(all_pred_records).to_csv("../data/procdata/1_doublingRate/MultiModel_All_Predictions.csv", index=False)

# Average the Gini importances (handling genes that might not be in every HVG set)
rf_imp_df = pd.concat(all_rf_importances, axis=1).mean(axis=1).sort_values(ascending=False)
rf_imp_df.to_csv("../data/procdata/1_doublingRate/Full_Ranked_Genes_for_Enrichment.csv", header=["Gini_Importance"])

print("\nDone! Results are now perfectly aligned with ModelZoo logic.")
