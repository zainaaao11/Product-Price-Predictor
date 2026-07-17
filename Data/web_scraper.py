from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

# ─── Settings ───
CATEGORIES = {
    "Téléphone & Tablette": "https://www.jumia.ma/telephone-tablette/",
    "TV & High Tech":       "https://www.jumia.ma/television-video/",
    "Electroménager":       "https://www.jumia.ma/electromenager/",
    "Beauté & Santé":       "https://www.jumia.ma/beaute-sante/",
    "Mode":                 "https://www.jumia.ma/vetements-chaussures/",
    "Maison & Bureau":      "https://www.jumia.ma/maison-cuisine-bureau/",
    "Sport & Loisirs":      "https://www.jumia.ma/sports-loisirs/",
    "Jeux Vidéos":          "https://www.jumia.ma/jeux-videos-consoles/",
    "Bébé & Jouets":        "https://www.jumia.ma/bebe-jouets/",
    "Accessoires Auto":     "https://www.jumia.ma/accessoire-auto-moto/"
    
}

PAGES_PER_CATEGORY = 30
WAIT_TIME = 10  

# ─── Price Cleaner ───
def clean_price(price_text):
    """Handles both single prices and price ranges like '73.47 - 185.07'"""
    price_text = price_text.replace("Dhs", "").replace(",", "").strip()
    if "-" in price_text:
        price_text = price_text.split("-")[0].strip()  # Take lowest price
    return float(price_text)

# ─── Load Existing Data ──
if os.path.exists("jumia_dataset.csv"):
    existing_df = pd.read_csv("jumia_dataset.csv")
    all_products = existing_df.to_dict("records")
    print(f"Loaded {len(all_products)} existing products")
else:
    all_products = []
    print("Starting fresh...")

# ─── Setup Edge ───
options = webdriver.EdgeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Edge(
    service=Service(EdgeChromiumDriverManager().install()),
    options=options
)

# ─── Scraper Function ───
def scrape_page(url, category_name):
    try:
        driver.get(url)
        time.sleep(WAIT_TIME)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        products = []
        cards = soup.find_all("article", class_="prd")

        for card in cards:
            try:
                a_tag = card.find("a", class_="core")
                if not a_tag:
                    continue

                name     = a_tag.get("data-ga4-item_name", "").strip()
                brand    = a_tag.get("data-ga4-item_brand", "").strip()
                subcat   = a_tag.get("data-ga4-item_category3", "").strip()
                price    = card.find(class_="prc")
                discount = card.find(class_="bdg _dsct")

                price_text    = price.text.strip() if price else None
                discount_text = discount.text.strip() if discount else "0%"

                if not price_text or not name:
                    continue

                price_clean = clean_price(price_text)

                products.append({
                    "name":     name,
                    "brand":    brand,
                    "category": category_name,
                    "subcat":   subcat,
                    "discount": discount_text,
                    "price":    price_clean,
                })

            except Exception as e:
                print(f"Skipped a product: {e}")
                continue

        return products

    except Exception as e:
        print(f"Page error: {e}")
        print(f"Waiting 10 seconds then retrying...")
        time.sleep(10)
        try:
            driver.get(url)
            time.sleep(WAIT_TIME)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            return []
        except:
            print(f"Failed again, skipping page")
            return []

# ─── Main Loop ──
for category_name, base_url in CATEGORIES.items():
    print(f"\nScraping: {category_name}")

    for page in range(1, PAGES_PER_CATEGORY + 1):
        url = f"{base_url}?page={page}"
        print(f"  Page {page}/{PAGES_PER_CATEGORY} — {len(all_products)} products so far")

        page_products = scrape_page(url, category_name)

        # ── Retry once if empty ──────────────────────────
        if not page_products:
            print(f"Empty page, retrying in 8 seconds...")
            time.sleep(8)
            page_products = scrape_page(url, category_name)

            if not page_products:
                print(f"Still empty, moving to next category")
                break

        all_products.extend(page_products)

        # ── Save after every page ──
        df = pd.DataFrame(all_products)
        df.drop_duplicates(subset=["name", "price"], inplace=True)
        df.to_csv("jumia_dataset.csv", index=False, encoding="utf-8-sig")

        time.sleep(0.35)

driver.quit()

# ─── Final Save ───
df = pd.DataFrame(all_products)
df.drop_duplicates(subset=["name", "price"], inplace=True)
df.to_csv("jumia_dataset.csv", index=False, encoding="utf-8-sig")

print(f"\nDone! {len(df)} products saved to jumia_dataset.csv")