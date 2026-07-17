import pandas as pd

#Additional exploration of the data 
df = pd.read_csv("jumia_dataset_final.csv", encoding="utf-8-sig", keep_default_na=False)
# Minimum price
min_price = df["price"].min()

# Maximum price
max_price = df["price"].max()

# Average (mean) price
mean_price = df["price"].mean()

#Median 
median= df["price"].median()
#standard deviation
std= df["price"].std()
print("standard deviation: ", std)
print("Minimum price:", min_price)
print("Maximum price:", max_price)
print("Average price:", mean_price)
print("Median price: ", median)