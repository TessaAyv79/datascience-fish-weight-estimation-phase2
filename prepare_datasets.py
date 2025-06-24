import os
import pandas as pd
import requests
from urllib.parse import urlparse
import yaml

CSV_PATH = "/home/tessaayv/datascience-weight-estimation/TheFishProject4/data/squid_dataset.csv"
OUTPUT_DIR = "datasets"
SPLIT_RATIO = 0.8  # %80 train, %20 val

# --- Sınıflandırma fonksiyonları ---
def weight_class(w): return str(min(int(w) // 20, 9))
def length_class(l): return str(min(max((int(l) - 3) // 4, 0), 5))
def color_class(c): return {'light': '0', 'medium': '1', 'dark': '2'}[c.lower()]
def species_class(s): return {'Loligo': '0', 'Illex': '1'}[s]

class_names = {
    "weight": [f"{i*20}-{(i+1)*20}g" for i in range(10)],
    "length": [f"{3+i*4}-{3+(i+1)*4}cm" for i in range(6)],
    "color": ["light", "medium", "dark"],
    "species": ["Loligo", "Illex"]
}

def download_image(url, dst_path):
    if pd.isna(url) or "undefined" in url or not url.startswith(("http://", "https://")):
        return False
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(dst_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"HTTP {response.status_code} for {url}")
            return False
    except Exception as e:
        print(f"⚠️ Download error for {url} - {e}")
        return False

# --- Veri yükleme ve temizleme ---
df = pd.read_csv(CSV_PATH)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Gerekli sütunlar
required_cols = ["image", "total_weight_(g)", "total_length_(cm)", "color", "species"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV dosyasında '{col}' sütunu eksik!")

# Yeniden adlandırma
df = df.rename(columns={
    "image": "filename",
    "total_weight_(g)": "weight",
    "total_length_(cm)": "length"
})

df = df.dropna(subset=["filename", "weight", "length", "color", "species"])

# Karışıklığı önlemek için url ön ekleri varsa temizle (örnek: 'data/processed/images/' ile başlayan satırlar)
def clean_url(url):
    if url.startswith("data/processed/images/"):
        return url[len("data/processed/images/"):]
    return url

df['filename'] = df['filename'].apply(clean_url)

# Karışıklık varsa başında boşluklar, yeni satırlar da temizlenebilir:
df['filename'] = df['filename'].str.strip()

# --- Dataset hazırlama fonksiyonu ---
def prepare_dataset(task, class_func, class_labels):
    print(f"\n📁 Preparing dataset for: {task}")
    task_dir = os.path.join(OUTPUT_DIR, task)
    for split in ['train', 'val']:
        for cls in range(len(class_labels)):
            os.makedirs(os.path.join(task_dir, split, str(cls)), exist_ok=True)

    n = len(df)
    for idx, row in df.iterrows():
        split = 'train' if idx < n * SPLIT_RATIO else 'val'
        try:
            cls = class_func(row[task])
            url = row["filename"]
            filename = os.path.basename(urlparse(url).path)
            dst_path = os.path.join(task_dir, split, cls, filename)

            if not os.path.exists(dst_path):
                if not download_image(url, dst_path):
                    print(f"🚫 Image not downloaded: {url}")
                    with open("missing_images.txt", "a") as f:
                        f.write(url + "\n")
        except Exception as e:
            print(f"⚠️ Skipped {row['filename']} - {e}")

    # YAML dosyası oluştur
    yaml_dict = {
        "train": f"{task_dir}/train",
        "val": f"{task_dir}/val",
        "nc": len(class_labels),
        "names": class_labels
    }
    with open(os.path.join(task_dir, "data.yaml"), 'w') as f:
        yaml.dump(yaml_dict, f)
    print(f"✅ {task}/data.yaml created.")

# --- Tüm görevleri sırayla çalıştır ---
prepare_dataset("weight", weight_class, class_names["weight"])
prepare_dataset("length", length_class, class_names["length"])
prepare_dataset("color", color_class, class_names["color"])
prepare_dataset("species", species_class, class_names["species"])
