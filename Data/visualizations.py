import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#plotting the price with the brand as a counterexample for non-linearity of the dataset
df = pd.read_csv("jumia_dataset_final.csv", encoding="utf-8-sig", keep_default_na=False)

# Get top 15 brands by count for readability
top_brands = df["brand"].value_counts().head(15).index
df_top = df[df["brand"].isin(top_brands)]

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_top, x="brand", y="price")
plt.xticks(rotation=45, ha="right")
plt.title("Price Distribution per Brand")
plt.xlabel("Brand")
plt.ylabel("Price (MAD)")
plt.tight_layout()
plt.show()