import pandas as pd

samples=[]
with open("../../data/rawdata/samples.txt") as fh:
    samples = [l.strip() for l in fh if l.strip()]

df = pd.read_excel("../../data/rawdata/sample_info.xlsx", sheet_name="Sheet2")
df["sample_name_standard"] = df["sample_name_standard"].astype(str)

df["map"] = df["sample_name_standard"].isin(samples).map({True: "Y", False: ""})

# -----------------------------
# QC: find missing samples
# -----------------------------
df_samples = set(df["sample_name_standard"].astype(str))
txt_samples = set(samples)

missing = sorted(txt_samples - df_samples)

print("\n===== Samples in samples.txt but NOT in Excel =====")
for s in missing:
    print(s)

print(f"\nTotal missing: {len(missing)}")

# -----------------------------
# Save result
# -----------------------------
df = df[df["map"] == "Y"]
df.to_excel("../../data/procdata/sample_info_map.xlsx", index=False)
