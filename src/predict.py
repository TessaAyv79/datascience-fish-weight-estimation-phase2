# src/predict.py

import torch
from torchvision import transforms, models
from PIL import Image
import os

MODEL_PATH = "models/model.pth"

def load_model():
    model = models.mobilenet_v2()
    # **DEĞİŞİKLİK**: Sınıf sayısı yerine çıktı nöronunu 1 yapıyoruz
    model.classifier[1] = torch.nn.Linear(model.last_channel, 1)
    
    # Modelin varlığını kontrol et
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    model = load_model()
    with torch.no_grad():
        output = model(image)
    
    # **DEĞİŞİKLİK**: Sınıf indeksi yerine doğrudan model çıktısını (ağırlık) döndürüyoruz
    predicted_weight = output.item()
    return predicted_weight

if __name__ == "__main__":
    # Örnek bir resimle test etmek için (önce data/processed klasörünüzde resim olduğundan emin olun)
    try:
        # data/processed klasöründeki ilk resmi alıp test edelim
        sample_image_dir = "data/processed"
        sample_image_name = os.listdir(sample_image_dir)[0]
        sample_image_path = os.path.join(sample_image_dir, sample_image_name)
        
        weight = predict(sample_image_path)
        print(f"Predicted weight for {sample_image_name}: {weight:.2f}g")
    except (FileNotFoundError, IndexError) as e:
        print(f"Cannot run prediction test: {e}")