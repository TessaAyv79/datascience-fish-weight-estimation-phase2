import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class SquidDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # 1. CSV'yi oku ve temel eksik verileri temizle
        df = pd.read_csv(csv_file)
        df.dropna(subset=['Image', 'Total Weight (g)'], inplace=True)

        # 2. 'Image' sütununun geçerli bir resim yolu içerdiğinden emin ol
        #    Sadece .jpg, .jpeg, .png ile biten satırları tut
        df = df[df['Image'].astype(str).str.contains('.jpg|.jpeg|.png', case=False, na=False)]

        # 3. Geçerli URL'lerden dosya adlarını oluştur
        df['ImageName'] = df['Image'].apply(lambda url: str(url).split('/')[-1])

        # 4. Sadece diskte gerçekten var olan işlenmiş resimleri filtrele
        existing_images = set(os.listdir(img_dir))
        self.labels_df = df[df['ImageName'].isin(existing_images)]

        print(f"Found {len(self.labels_df)} valid and existing images for training.")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx]['ImageName'])
        image = Image.open(img_name).convert("RGB")

        weight = self.labels_df.iloc[idx]['Total Weight (g)']
        weight = torch.tensor([weight], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, weight
