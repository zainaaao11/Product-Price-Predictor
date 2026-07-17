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
df = pd.read_csv("../All_About_The_Data/jumia_dataset_clean.csv", encoding="utf-8-sig")

# Check if we already have product_name column from a previous run
if "product_name" not in df.columns:
    df["product_name"] = ""
    df["product_details"] = ""

print(f"Total rows: {len(df)}")

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
                "content": f"""Extract product name and details from each product title below.
Return ONLY a JSON array, nothing else:
[
  {{"product_name": "...", "product_details": "..."}},
  ...
]

Product titles:
{numbered}"""
            }]
        }
    )
    
    text = response.json()["choices"][0]["message"]["content"].strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

# ─── Main Loop ─
BATCH_SIZE = 10

for i in range(0, len(df), BATCH_SIZE):
    # Skip already processed rows
    if df.at[i, "product_name"] != "":
        print(f"Skipping already processed row {i}")
        continue
    
    batch = df["name"].iloc[i:i+BATCH_SIZE].tolist()
    batch_indices = df.index[i:i+BATCH_SIZE].tolist()
    
    try:
        results = extract_batch(batch)
        
        for j, idx in enumerate(batch_indices):
            df.at[idx, "product_name"] = results[j].get("product_name", "")
            df.at[idx, "product_details"] = results[j].get("product_details", "")
        
        print(f"Processed {min(i+BATCH_SIZE, len(df))}/{len(df)} products")
        time.sleep(1.5)

    except Exception as e:
        print(f"Error: {e}")
        print(f"Waiting 60 seconds...")
        time.sleep(60)
        try:
            results = extract_batch(batch)
            for j, idx in enumerate(batch_indices):
                df.at[idx, "product_name"] = results[j].get("product_name", "")
                df.at[idx, "product_details"] = results[j].get("product_details", "")
        except:
            pass

    # ── Save every 100 rows ───
    if i % 100 == 0 and i > 0:
        df.to_csv("jumia_dataset_final.csv", index=False, encoding="utf-8-sig")
        print(f"Progress saved at row {i}")

# ─── Final Save ──────
df.to_csv("jumia_dataset_final.csv", index=False, encoding="utf-8-sig") 
print(f"\nDone! {len(df)} products saved")