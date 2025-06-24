# src/create_fixed_csv.py
import os
import pandas as pd

img_dir = 'data/processed'
records = []

for root, _, files in os.walk(img_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            relative_path = os.path.relpath(os.path.join(root, file), img_dir)
            label = 0.5  # Bu örnek. Gerçek ağırlık etiketlerini squid_dataset.csv'den eşleştirerek koyabiliriz.
            records.append([relative_path, label])

df = pd.DataFrame(records, columns=['filename', 'label'])
df.to_csv('data/fixed_dataset.csv', index=False)
print("✅ CSV dosyası oluşturuldu: data/fixed_dataset.csv")
