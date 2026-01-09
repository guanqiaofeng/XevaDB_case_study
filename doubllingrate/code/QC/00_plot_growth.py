import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# output directory
os.makedirs("../../data/QC/growth_curve_plots", exist_ok=True)

# Load data
df = pd.read_csv("../../data/clean/experiment_clean.csv")

# Extract replicate ID (e.g. m1, m2, ...)
df["replicate"] = df["model.id"].str.extract(r"\.(m\d+)$")

# Choose the model to plot
model = "REF036"

# Filter data for that model
sub = df[df["modelID"] == model].copy()

# Sort by time for each replicate
sub = sub.sort_values(["replicate", "time"])

# Set a nice color palette
sns.set_style("whitegrid")
palette = sns.color_palette("tab10", n_colors=sub["replicate"].nunique())

plt.figure(figsize=(6,4))

# Plot one line per replicate
for i, (rep, grp) in enumerate(sub.groupby("replicate")):
    plt.plot(grp["time"], grp["volume"], marker="o",
             label=rep, color=palette[i])

plt.title(f"Growth curves for {model}")
plt.xlabel("Time (days)")
plt.ylabel("Tumor volume (mm³)")
plt.legend(title="Replicate", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
