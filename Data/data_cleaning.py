import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv("jumia_dataset.csv", encoding="utf-8-sig")
print(f"Loaded {len(df)} products")

#Standardize brand casing
df["brand"] = df["brand"].str.strip().str.title()

#Strip whitespace from categories
df["main_category"] = df["category"].str.strip()
df["sub_category"] = df["subcat"].str.strip()

#Remove price outliers (per category)
before = len(df)
rows_to_keep = []
for cat, group in df.groupby("main_category"):
    mean = group["price"].mean()
    std = group["price"].std()
    filtered = group[(group["price"] >= 10) & (group["price"] <= mean + 3 * std)]
    rows_to_keep.append(filtered)

df = pd.concat(rows_to_keep).reset_index(drop=True)
print(f"Removed {before - len(df)} outlier rows")

#Add log price
df["log_price"] = np.log1p(df["price"])

#Clean name
df["name"] = df["name"].str.strip()

#Drop discount (supervisor said not needed)
df = df.drop(columns=["discount"])

#Drop subCat (not needed since most of the time it's similar to category)
df = df.drop(columns=["subcat"])

#Reorder columns
df = df[["name", "brand", "main_category", "price", "log_price"]]

#discover the dataset 
print(f"\n=== FINAL DATASET ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nProducts per main category:")
print(df["main_category"].value_counts().to_string())
print(f"\nSample:")
print(df.head(3).to_string())

#save again 
df.to_csv("jumia_dataset_clean.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved as jumia_dataset_clean.csv")
