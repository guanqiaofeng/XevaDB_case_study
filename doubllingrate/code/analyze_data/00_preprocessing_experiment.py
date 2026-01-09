import pandas as pd
import numpy as np
from scipy.stats import linregress

# --- Parameters ---
MAX_DAY = 35      # use first 28 days to estimate exponential growth
MIN_POINTS = 3    # minimum number of time points per mouse
MIN_R2 = 0.5      # minimum fit quality

# --- Load data ---
df = pd.read_csv("../../data/clean/experiment_clean.csv")

# Keep positive volumes only
df = df[df["volume"] > 0]

# --- Fit log(volume) ~ time for each mouse (model.id) ---
per_mouse = []

for mouse_id, sub in df.groupby("model.id"):
    sub = sub[sub["time"] <= MAX_DAY].sort_values("time")
    if sub.shape[0] < MIN_POINTS:
        continue

    slope, intercept, r, p, se = linregress(sub["time"], np.log(sub["volume"]))
    r2 = r**2
    if slope <= 0 or r2 < MIN_R2:
        continue

    dt = np.log(2) / slope
    per_mouse.append({
        "mouse_id": mouse_id,
        "modelID": sub["modelID"].iloc[0],
        "growth_rate_k": slope,
        "doubling_time_days": dt,
        "r2": r2,
        "n_points": sub.shape[0],
        "max_day_used": sub["time"].max()
    })

per_mouse_df = pd.DataFrame(per_mouse)

# --- Aggregate to model level ---
per_model_df = (
    per_mouse_df.groupby("modelID")
    .agg(
        mean_doubling_time_days=("doubling_time_days", "mean"),
        sd_doubling_time_days=("doubling_time_days", "std"),
        mean_growth_rate_k=("growth_rate_k", "mean"),
        mean_r2=("r2", "mean"),
        n_mice=("mouse_id", "nunique")
    )
    .reset_index()
)

# --- Save results ---
per_mouse_df.to_csv("../../data/analyze/doubling_time_per_mouse.csv", index=False)
per_model_df.to_csv("../../data/analyze/doubling_time_per_model.csv", index=False)

print("✅ Finished computing doubling times")
print(per_model_df.head())
