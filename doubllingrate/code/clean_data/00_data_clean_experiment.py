import pandas as pd
import numpy as np

df = pd.read_csv("../../data/raw/experiment.csv", usecols=["model.id", "drug", "time", "volume", "modelID", "sampleID", "model.type"])

df = df[df["model.type"] == "Parental"]
df = df[df["drug"] == "H2O"]

print(df)

df.to_csv("../../data/process/experiment_parental_H20.csv", index=False)

# save model id 
model_list = sorted(df["modelID"].dropna().unique().astype(str))

pd.Series(model_list, name="modelID").to_csv("../../data/process/experiment_models.txt", index=False)

