"""
When extracting the data some rows might not be processed because of connexion timeouts so they will be skipped. Hence, this code processes those
rows and store them back to the final dataset
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
BATCH_SIZE = 10

original = pd.read_csv(
    "../FirstAttempt/jumia_dataset_final.csv",
    encoding="utf-8-sig",
    keep_default_na=False
)

processed = pd.read_csv(
    "jumia_dataset_final.csv",
    encoding="utf-8-sig",
    keep_default_na=False
)

print(f"Original rows : {len(original)}")
print(f"Processed rows: {len(processed)}")

# Find rows that still need processing

missing_rows = processed[
    (processed["name"] == "") |
    (processed["usage"] == "") |
    (processed["composition"] == "")
].index.tolist()

print(f"\nRemaining rows to process: {len(missing_rows)}")

if len(missing_rows) == 0:
    print("Everything is already processed.")
    exit()

# Extraction Function

def extract_batch(batch_titles):

    numbered = "\n".join(
        [f"{i+1}. {title}" for i, title in enumerate(batch_titles)]
    )

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

Return ONLY a JSON array with exactly this structure:

[
  {{
    "name": "short clean product name only (brand + model name)",
    "usage": "what this product is used for in one short sentence",
    "composition": "key specs and details such as size, power, capacity, color, material, gender etc"
  }}
]

Product names:

{numbered}
"""
            }]
        },
        timeout=120
    )

    response.raise_for_status()

    text = response.json()["choices"][0]["message"]["content"].strip()
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)

# Process only missing rows

for start in range(0, len(missing_rows), BATCH_SIZE):

    batch_indices = missing_rows[start:start + BATCH_SIZE]
    batch_titles = original.loc[batch_indices, "product_name"].tolist()

    success = False

    for attempt in range(3):

        try:

            results = extract_batch(batch_titles)

            for j, idx in enumerate(batch_indices):

                if j < len(results):
                    processed.at[idx, "name"] = results[j].get("name", "")
                    processed.at[idx, "usage"] = results[j].get("usage", "")
                    processed.at[idx, "composition"] = results[j].get("composition", "")

            success = True
            break

        except Exception as e:

            print(f"\nError in batch starting at row {batch_indices[0]}")
            print(e)

            if attempt < 2:
                print("Retrying in 20 seconds...")
                time.sleep(20)

    if not success:
        print(f"Skipped batch starting at row {batch_indices[0]}")

    # Save after every batch
    processed.to_csv(
        "jumia_dataset_final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Processed {min(start + BATCH_SIZE, len(missing_rows))}/{len(missing_rows)} remaining rows.")

    time.sleep(1.5)

# Final Save

processed.to_csv(
    "jumia_dataset_final.csv",
    index=False,
    encoding="utf-8-sig"
)

remaining = processed[
    (processed["name"] == "") |
    (processed["usage"] == "") |
    (processed["composition"] == "")
]

print("\n====================================")
print("Finished!")
print(f"Rows still missing: {len(remaining)}")
print("Dataset saved as: jumia_dataset_final.csv")
print("====================================")