import os
from PIL import Image

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def preprocess_images():
    # ## BU SATIR ÇOK ÖNEMLİ! ##
    # 'data/processed' klasörünü, eğer yoksa, oluşturur.
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # data/raw klasöründe resim olup olmadığını kontrol et
    if not os.listdir(RAW_DIR):
        print(f"Warning: Raw data directory '{RAW_DIR}' is empty. Nothing to preprocess.")
        return

    print(f"Processing images from '{RAW_DIR}'...")
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(RAW_DIR, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize((224, 224))
                    img.save(os.path.join(PROCESSED_DIR, filename))
            except Exception as e:
                print(f"Could not process {filename}: {e}")
    print("Image preprocessing complete.")

if __name__ == "__main__":
    preprocess_images()