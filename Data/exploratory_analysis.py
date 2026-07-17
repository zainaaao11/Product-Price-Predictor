import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("jumia_dataset_final.csv")
print(df.head()) #prints the first 5 rows
print(df.shape)  #prints a tuple of #rows and #columns
print(df.columns) #prints pandas.index object of the columns
print(df.info()) 
print(df["product_details"].isnull().sum())
print(df.isnull().sum()) #num of missing values in each column
#fill the missing values with an empty string 
df["product_details"]=df["product_details"].fillna("")
df.to_csv("jumia_dataset_final.csv", index=False, encoding="utf-8-sig")
df = pd.read_csv("jumia_dataset_final.csv", encoding="utf-8-sig", keep_default_na=False)
print(df.isnull().sum())
print(df.describe())
print(df.duplicated().sum())
dups=df[df.duplicated(keep=False)]
print(dups)
df=df.drop_duplicates()
df.to_csv("jumia_dataset_final.csv", index=False, encoding="utf-8-sig")
print(df.duplicated().sum())
print(df["brand"].unique())
print(df["main_category"].unique())
print(df["main_category"].value_counts())

#Some plotting
plt.hist(df["price"], bins=30)
plt.xlabel("Price")
plt.ylabel("Number of Products")
plt.title("Distribution of Prices")
plt.boxplot(df["price"])
plt.show()

#Exploratory analysis of the brand column for the One-hot-encoding
print(df["brand"].nunique()) #there are 838 unique brands
print(df["main_category"].nunique()) #there are 9 main cats

print(df["brand"].value_counts().head(20))  # most common brands
print(f"\nBrands appearing only once: {(df['brand'].value_counts() == 1).sum()}") 
print(f"Brands appearing less than 5 times: {(df['brand'].value_counts() < 5).sum()}")

#Keep only the brands that appear more than 20 times 
threshold = 20
frequent_brands = df["brand"].value_counts()
kept = frequent_brands[frequent_brands >= threshold]
print(f"Brands kept: {len(kept)}")
print(f"Products covered: {kept.sum()} / {len(df)}")
print(f"\nBrand names kept:")
print(kept)