# Product-Price-Prediction

A machine learning project that predicts product prices on **Jumia.ma** (Morocco's e-commerce platform) using machine learning models trained on textual and categorical product information.

## Overview

The project scrapes real product listings from Jumia, cleans and enriches the data, then trains regression models to predict price (in MAD) from product text and metadata. Two different modeling approaches were built and compared, with the second approach serving as the final model due to its improved feature representation and predictive performance.

## Project Structure

```
Product-Price-Prediction/
├── Data/     # Scraping, cleaning, and exploration
│   ├── web_scraper.py       # Collects product data from Jumia (Selenium + BeautifulSoup)
│   ├── data_cleaning.py          # Cleans raw data, removes outliers, log-transforms price
│   ├── exploratory_analysis.py   # Missing values, duplicates, distributions
│   ├── dataset_summary.py    # Summary statistics
│   └── visualizations.py               # Visualizations
│
├── Baseline_model/            # V1: TF-IDF + One-Hot Encoding + Random Forest
│   ├── extract_product_info.py
|   ├── check-after-extraction.py 
│   ├── feature_encoding.py
│   └── train_random_forest.py
│
├── Embedding_model/           # V2: Text Embeddings + Target Encoding + XGBoost
|   ├── install-bge-model.py
│   ├── extract_product_info.py
|   ├── check-after-extraction.py
│   ├── embedding_encoding.py
│   ├── train_xgboost.py
│   └── hyperparameter_tuning.py
│
└── requirements.txt
```

## Pipeline

1. **Scrape** — Collect product listings (name, brand, category, price) across 10 Jumia categories.
2. **Clean** — Standardize fields, remove price outliers per category, log-transform the price.
3. **Explore** — Check distributions, duplicates, and brand/category frequency.
4. **Extract** — Use the Mistral AI API to extract structured information (name, usage, composition, etc.) from unstructured product names.
5. **Encode** — Turn text and categorical fields into numeric features.
6. **Train & Evaluate** — Fit a regression model on `log_price` and evaluate on held-out test data (RMSE, MAE, R²).

## The Two Attempts

**First Attempt (Baseline-model):** Product text is split into `product_name` / `product_details` via the Mistral API, then vectorized with TF-IDF. Brand and category are one-hot encoded. A **Random Forest Regressor** is trained on the combined features.

**Second Attempt (Embedding-model):** Product text is split more granularly into `name` / `usage` / `composition`, then embedded using the **BGE-M3** sentence transformer instead of TF-IDF. Brand and category use target (mean-price) encoding instead of one-hot. An **XGBoost Regressor** is trained on the combined features, with `hyperparameterTuning.py` used to search for the best parameters.

## Setup

-Python 3.12 or later

Install the required packages:

```bash
pip install -r requirements.txt
```

To re-run the extraction steps (`extract_product_info.py`), you'll need a Mistral AI API key in a `.env` file:

```
MISTRAL_API_KEY=your_key_here
```

To re-run the scraper, a Microsoft Edge browser installation is required (handled automatically via `webdriver-manager`).

The BGE-M3 embedding model is downloaded automatically from Hugging Face the first time it is used.

## Technologies Used 

Python · Selenium · BeautifulSoup · pandas · scikit-learn · XGBoost · Sentence Transformers (BGE-M3) · Mistral AI

---
For detailed methodology, results, and analysis, see the accompanying project report.
