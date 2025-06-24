import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import SquidDataset
import os

# Sabitler
DATA_DIR = "data/processed"
CSV_PATH = "data/squid_dataset.csv"
MODEL_PATH = "models/model.pth"
BATCH_SIZE = 16
EPOCHS = 10

# Bozuk örnekleri filtreleyen özel collate fonksiyonu
def custom_collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SquidDataset(csv_file=CSV_PATH, img_dir=DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 1)  # Regresyon için çıktı 1
    model = model.to(device)

    criterion = nn.MSELoss()  # Regresyon için kayıp fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training started...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch in dataloader:
            if batch is None:
                continue  # Tüm batch bozuksa atla

            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
