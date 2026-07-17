import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Load Data ─────
df = pd.read_csv("jumia_dataset_final.csv", encoding="utf-8-sig", keep_default_na=False)
print(f"Dataset shape: {df.shape}")

# ───Split ────
X = df[["name", "usage", "composition", "brand", "main_category"]] #features 
y = df["log_price"]  #target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ───Target Encoding ────
brand_means = X_train.copy()
brand_means["log_price"] = y_train
brand_encoding = brand_means.groupby("brand")["log_price"].mean()

cat_means = X_train.copy()
cat_means["log_price"] = y_train
cat_encoding = cat_means.groupby("main_category")["log_price"].mean()

global_mean = y_train.mean()

X_train["brand_encoded"]    = X_train["brand"].map(brand_encoding).fillna(global_mean)
X_train["category_encoded"] = X_train["main_category"].map(cat_encoding).fillna(global_mean)

X_test["brand_encoded"]    = X_test["brand"].map(brand_encoding).fillna(global_mean)
X_test["category_encoded"] = X_test["main_category"].map(cat_encoding).fillna(global_mean)

print(f"Target Encoding done")

# ───Load BGE-M3 ────
print(f"\nLoading BGE-M3 model...")
embedding_model = SentenceTransformer("BAAI/bge-m3")
print(f"BGE-M3 loaded!")

# ───Embed and Save Function ────
def embed_and_save(texts, filename, model):
    # If already saved, just load it (crash protection)
    if os.path.exists(filename):
        print(f"Already exists, loading {filename}")
        return np.load(filename)
    
    print(f"Embedding {filename}...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Save immediately after encoding
    np.save(filename, embeddings)
    print(f"Saved {filename}! Shape: {embeddings.shape}")
    return embeddings

# ───Embed All Text Columns ───
train_name_emb        = embed_and_save(X_train["name"].tolist(),        "train_name_emb.npy",        embedding_model)
train_usage_emb       = embed_and_save(X_train["usage"].tolist(),       "train_usage_emb.npy",       embedding_model)
train_composition_emb = embed_and_save(X_train["composition"].tolist(), "train_composition_emb.npy", embedding_model)

test_name_emb        = embed_and_save(X_test["name"].tolist(),        "test_name_emb.npy",        embedding_model)
test_usage_emb       = embed_and_save(X_test["usage"].tolist(),       "test_usage_emb.npy",       embedding_model)
test_composition_emb = embed_and_save(X_test["composition"].tolist(), "test_composition_emb.npy", embedding_model)

# ───Combine All Features ────
X_train_final = np.hstack([
    train_name_emb,
    train_usage_emb,
    train_composition_emb,
    X_train[["brand_encoded", "category_encoded"]].values
])

X_test_final = np.hstack([
    test_name_emb,
    test_usage_emb,
    test_composition_emb,
    X_test[["brand_encoded", "category_encoded"]].values
])

print(f"\nFinal feature matrix:")
print(f"   Train: {X_train_final.shape}")
print(f"   Test:  {X_test_final.shape}")

# ───Save Everything ────
np.save("X_train_final.npy", X_train_final)
np.save("X_test_final.npy", X_test_final)
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(brand_encoding, "brand_encoding.pkl")
joblib.dump(cat_encoding, "cat_encoding.pkl")
joblib.dump(global_mean, "global_mean.pkl")

print(f"\nAll files saved! Ready for model training!")