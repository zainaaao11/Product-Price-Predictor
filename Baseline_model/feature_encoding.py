import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import scipy.sparse as sp
import joblib
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Load Data ───
df = pd.read_csv("jumia_dataset_final.csv", encoding="utf-8-sig", keep_default_na=False) 

# ───Reduce brands to top 62 + Other ───
top_brands = df["brand"].value_counts()
top_brands = top_brands[top_brands >= 20].index
df["brand"] = df["brand"].apply(lambda x: x if x in top_brands else "Other")

# ───Define Features and Label ────
X = df[["product_name", "product_details", "brand", "main_category"]]
y = df["log_price"]

# ───Split ────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ───One Hot Encode ───
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
cat_train = encoder.fit_transform(X_train[["brand", "main_category"]])
cat_test  = encoder.transform(X_test[["brand", "main_category"]])

# ───TF-IDF product_name ────
tfidf_name = TfidfVectorizer(max_features=100)
name_train = tfidf_name.fit_transform(X_train["product_name"])
name_test  = tfidf_name.transform(X_test["product_name"])

# ───TF-IDF product_details ───
tfidf_details = TfidfVectorizer(max_features=100)
details_train = tfidf_details.fit_transform(X_train["product_details"])
details_test  = tfidf_details.transform(X_test["product_details"])

# ───Combine ────
X_train_final = hstack([cat_train, name_train, details_train])
X_test_final  = hstack([cat_test,  name_test,  details_test])

print(f"X_train_final shape: {X_train_final.shape}")
print(f"X_test_final shape: {X_test_final.shape}")

# ───Save Everything ───
joblib.dump(encoder,       "encoder.pkl")
joblib.dump(tfidf_name,    "tfidf_name.pkl")
joblib.dump(tfidf_details, "tfidf_details.pkl")
joblib.dump(y_train,       "y_train.pkl")
joblib.dump(y_test,        "y_test.pkl")

# ─── Save Matrices────
sp.save_npz("X_train_final.npz", X_train_final)
sp.save_npz("X_test_final.npz", X_test_final)

print("All files saved successfully!")