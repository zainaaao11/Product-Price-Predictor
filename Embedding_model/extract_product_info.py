import requests
import pandas as pd
import json
import time
import os 
from dotenv import load_dotenv
from mistralai import Mistral
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#read the API key from the .env file
load_dotenv()
api_key= os.getenv("MISTRAL_API_KEY")

if api_key is None:
    raise ValueError("MISTRAL_API_KEY not found.")

client=Mistral(api_key=api_key)

#load the dataset

df = pd.read_csv("../FirstAttempt/jumia_dataset_final.csv", encoding="utf-8-sig", keep_default_na=False)

print(f"Total rows: {len(df)}")

# Add new columns if they don't exist
if "extracted_name" not in df.columns:
    df["extracted_name"] = "" 
    df["usage"] = ""
    df["composition"] = ""

def extract_batch(batch_titles):
    numbered = "\n".join([f"{i+1}. {title}" for i, title in enumerate(batch_titles)])
    
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-small-latest",
            "messages": [{
                "role": "user",
                "content": f"""For each product name below, extract three pieces of information.
Return ONLY a JSON array with exactly this structure, nothing else:
[
  {{
    "name": "short clean product name only (brand + model name)",
    "usage": "what this product is used for in one short sentence",
    "composition": "key specs and details such as size, power, capacity, color, material, gender etc"
  }},
  ...
]

Product names:
{numbered}"""
            }]
        }
    )
    
    text = response.json()["choices"][0]["message"]["content"].strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

# ─── Main Loop ────
BATCH_SIZE = 10

for i in range(0, len(df), BATCH_SIZE):
    # Skip already processed rows
    if df.at[i, "extracted_name"] != "":
        if i % 500 == 0:
            print(f"Skipping already processed rows up to {i}")
        continue

    batch_indices = df.index[i:i+BATCH_SIZE].tolist()
    batch_titles = df.loc[batch_indices, "product_name"].tolist()

    try:
        results = extract_batch(batch_titles)

        for j, idx in enumerate(batch_indices):
            if j < len(results):
                df.at[idx, "extracted_name"] = results[j].get("name", "")
                df.at[idx, "usage"]          = results[j].get("usage", "")
                df.at[idx, "composition"]    = results[j].get("composition", "")
            else:
                df.at[idx, "extracted_name"] = batch_titles[j]
                df.at[idx, "usage"]          = ""
                df.at[idx, "composition"]    = ""

        print(f"Processed {min(i+BATCH_SIZE, len(df))}/{len(df)} products")
        time.sleep(1.5)

    except Exception as e:
        print(f"Error at batch {i}: {e}")
        print(f"Waiting 60 seconds...")
        time.sleep(60)

        try:
            results = extract_batch(batch_titles)
            for j, idx in enumerate(batch_indices):
                if j < len(results):
                    df.at[idx, "extracted_name"] = results[j].get("name", "")
                    df.at[idx, "usage"]          = results[j].get("usage", "")
                    df.at[idx, "composition"]    = results[j].get("composition", "")
                else:
                    df.at[idx, "extracted_name"] = batch_titles[j]
                    df.at[idx, "usage"]          = ""
                    df.at[idx, "composition"]    = ""
        except:
            for idx in batch_indices:
                df.at[idx, "extracted_name"] = ""
                df.at[idx, "usage"]          = ""
                df.at[idx, "composition"]    = ""

    # ── Save every 100 rows ───
    if i % 100 == 0 and i > 0:
        df.to_csv("jumia_dataset_final.csv", index=False, encoding="utf-8-sig")
        print(f"Progress saved at row {i}")

# ─── Final cleanup ───
# Drop old columns and rename new ones
df = df.drop(columns=["product_name", "product_details"])
df = df.rename(columns={"extracted_name": "name"})

# Reorder columns
df = df[["name", "usage", "composition", "brand",
         "main_category",
         "price", "log_price"]]

# ─── Final Save ─────
df.to_csv("jumia_dataset_final.csv", index=False, encoding="utf-8-sig")
print(f"\nDone! Dataset saved with new columns:")
print(df.head(5).to_string())