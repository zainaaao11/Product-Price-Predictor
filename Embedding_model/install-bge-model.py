from sentence_transformers import SentenceTransformer

# This will download BGE-M3 from HuggingFace (first time only, ~2GB)
model = SentenceTransformer("BAAI/bge-m3")

# Test with one product
embedding = model.encode("Samsung Galaxy A55 128GB")
print(f"Embedding shape: {embedding.shape}")  # Should print (1024,)
print("BGE-M3 working!")