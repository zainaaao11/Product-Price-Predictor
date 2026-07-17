"""
in the InfoExtractionVer1 file, some rows haven't been processed and skipped because of timeout.
Thus, in this file we will be handling those unprocessed rows 
"""
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
df = pd.read_csv("jumia_dataset_final.csv", encoding="utf-8-sig", keep_default_na=False)

# Add columns if they don't exist
if "product_name" not in df.columns:
    df["product_name"] = ""
    df["product_details"] = ""

# ─── Check unprocessed rows ────
unprocessed = df[df["product_name"] == ""]
print(f"Total rows: {len(df)}")
print(f"Already processed: {len(df) - len(unprocessed)}")
print(f"Still need processing: {len(unprocessed)}")

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

# ─── Process Only Unprocessed Rows ───
BATCH_SIZE = 10
unprocessed_indices = unprocessed.index.tolist()

for i in range(0, len(unprocessed_indices), BATCH_SIZE):
    batch_indices = unprocessed_indices[i:i+BATCH_SIZE]
    batch_titles  = df.loc[batch_indices, "name"].tolist()
    
    try:
        results = extract_batch(batch_titles)
        
        for j, idx in enumerate(batch_indices):
            if j < len(results):
                df.at[idx, "product_name"]    = results[j].get("product_name", "")
                df.at[idx, "product_details"] = results[j].get("product_details", "")
            else:
                # If Mistral returned fewer results keep original name
                df.at[idx, "product_name"]    = batch_titles[j]
                df.at[idx, "product_details"] = ""
        
        print(f"Processed {min(i+BATCH_SIZE, len(unprocessed_indices))}/{len(unprocessed_indices)} unprocessed rows")
        time.sleep(1.5)

    except Exception as e:
        print(f"Error: {e}")
        print(f"Waiting 60 seconds...")
        time.sleep(60)
        
        try:
            results = extract_batch(batch_titles)
            for j, idx in enumerate(batch_indices):
                if j < len(results):
                    df.at[idx, "product_name"]    = results[j].get("product_name", "")
                    df.at[idx, "product_details"] = results[j].get("product_details", "")
                else:
                    df.at[idx, "product_name"]    = batch_titles[j]
                    df.at[idx, "product_details"] = ""
        except:
            # If still fails keep original name
            for idx in batch_indices:
                df.at[idx, "product_name"]    = ""
                df.at[idx, "product_details"] = ""

    # ── Save every 100 rows ───
    if i % 100 == 0 and i > 0:
        df.to_csv("../All_About_The_Data/jumia_dataset_clean.csv", index=False, encoding="utf-8-sig")
        print(f"Progress saved at row {i}")

df = df.drop(columns=["name"])

#Reorder columns
df = df[["product_name", "product_details", "brand", "main_category", "price", "log_price"]]

# ─── Final Save ──
df.to_csv("jumia_dataset_final.csv", index=False, encoding="utf-8-sig")
print(f"\nDone! {len(df)} products saved")