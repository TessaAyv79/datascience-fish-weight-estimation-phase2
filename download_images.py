import pandas as pd
import requests
import os
from tqdm import tqdm
# Dosya yollarını tanımla
CSV_PATH = "data/squid_dataset.csv"
RAW_DIR = "data/raw"

def download_images_from_csv():
    # Çıktı klasörünü oluştur
    os.makedirs(RAW_DIR, exist_ok=True)

    # CSV dosyasını oku
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: The file {CSV_PATH} was not found.")
        return

    if 'Image' not in df.columns:
        print(f"Error: 'Image' column not found in {CSV_PATH}.")
        return

    print(f"Found {len(df)} image URLs in the CSV file. Starting download...")

    # Her bir URL için resmi indir
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading Images"):
        url = row['Image']
        if pd.isna(url) or not isinstance(url, str):
            print(f"Warning: Skipping invalid URL at row {index}")
            continue
        
        try:
            filename = url.split('/')[-1]
            save_path = os.path.join(RAW_DIR, filename)
            
            # Eğer dosya zaten varsa indirme
            if os.path.exists(save_path):
                continue

            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                print(f"Warning: Failed to download {url}. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred for URL {url}: {e}")

    print("Image download process complete.")

if __name__ == "__main__":
    download_images_from_csv()
