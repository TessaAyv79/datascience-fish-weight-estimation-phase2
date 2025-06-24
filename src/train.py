import os
from ultralytics import YOLO # YOLO modelini import ediyoruz
import yaml # data.yaml dosyasını okumak için
import torch # GPU kontrolü ve genel PyTorch işlemleri için

# --- Sabitler ---
# Bu DATA_YAML_PATH, sizin veri setinizin ve sınıf bilgilerinizin olduğu data.yaml dosyasına işaret etmelidir.
# Proje yapınıza göre örnek yollar:
# - Cilt durumu sınıflandırması için: '2_squid/4b_classification(skin_status)/1_train_test_split/data.yaml'
# - Tür ve Renk sınıflandırması için: '2_squid/4a_classification(species&colour)/1_train_test_split/data.yaml'
DATA_YAML_PATH = '2_squid/4b_classification(skin_status)/1_train_test_split/data.yaml' # Lütfen bu yolu kendi projenize göre düzeltin!
MODEL_NAME = 'yolov8n-cls.pt' # Başlangıç ağırlıkları (nano sınıflandırma modeli)
PROJECT_DIR = 'runs/classify' # Eğitim sonuçlarının kaydedileceği ana klasör
EXP_NAME = 'skin_status_yolo_training' # Bu eğitim run'ının ismi (klasör adı olacak)
EPOCHS = 100 # Eğitim döngüsü sayısı (MobileNet'teki 10'dan daha fazla olabilir)
IMG_SIZE = 224 # Görüntü boyutu (YOLO genellikle 224, 640 gibi boyutları kullanır)
BATCH_SIZE = 16 # Batch boyutu

def train_yolo_classification():
    """
    YOLOv8 Sınıflandırma Modelini Eğitir.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # data.yaml dosyasından sınıf bilgilerini al
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            data_config = yaml.safe_load(f)
        num_classes = len(data_config['names'])
        print(f"Data YAML dosyası yüklendi. Sınıf sayısı: {num_classes}")
    except FileNotFoundError:
        print(f"HATA: {DATA_YAML_PATH} bulunamadı. Lütfen yolu kontrol edin.")
        return
    except Exception as e:
        print(f"HATA: Data YAML dosyası okunurken sorun oluştu: {e}")
        return

    # YOLO modelini başlat
    # classification modelini yüklerken 'cls' modunda olmalı
    # Eğer önceden eğitilmiş bir sınıflandırma modeli yoksa, yolov8n.pt gibi detection modelinden başlayabiliriz,
    # ancak en uygunu 'yolov8n-cls.pt' gibi sınıflandırma için eğitilmiş bir modeldir.
    model = YOLO(MODEL_NAME) # yolov8n-cls.pt veya yolov8m-cls.pt gibi bir model
    
    # Modelin sınıflandırma modunda olduğundan emin olun (varsayılan olarak zaten öyle olmalı)
    if model.task != 'classify':
        print("Uyarı: Model sınıflandırma görevi için değil. Doğru modeli yüklediğinizden emin olun (örn. yolov8n-cls.pt).")
        return

    print(f"YOLOv8 Sınıflandırma Eğitimi Başlatılıyor ({MODEL_NAME})...")
    
    # Model eğitimi
    results = model.train(
        data=DATA_YAML_PATH,          # Veri setinizin data.yaml yolu
        epochs=EPOCHS,                # Eğitim döngüsü sayısı
        imgsz=IMG_SIZE,               # Görüntü boyutu
        batch=BATCH_SIZE,             # Batch boyutu
        project=PROJECT_DIR,          # Eğitim sonuçlarının kaydedileceği ana klasör
        name=EXP_NAME,                # Bu eğitim run'ının ismi
        device=device,                # Kullanılacak cihaz (cuda veya cpu)
        # pretrained=True,            # Varsayılan olarak True'dur, önceden eğitilmiş ağırlıkları kullanır
        # patience=50,                # Erken durdurma sabrı (isteğe bağlı)
        # cache=True,                 # Verileri önbelleğe al (eğitim hızlandırabilir)
        # workers=os.cpu_count(),     # Veri yükleme için kullanılacak CPU çekirdeği sayısı
        # plots=True,                 # Eğitim metrik grafiklerini oluştur
    )

    print("\nEğitim Tamamlandı!")
    print(f"Sonuçlar şuraya kaydedildi: {model.trainer.save_dir}")
    print(f"En iyi model ağırlıkları: {os.path.join(model.trainer.save_dir, 'weights', 'best.pt')}")

    # Eğitim sonrası ek analizler veya görselleştirmeler yapılabilir.
    # Örneğin, sonuçlar nesnesinden eğitim metriklerine erişebilirsiniz.
    # print(results.metrics) # Eğitim sonrası metrikler (accuracy, loss vb.)

if __name__ == "__main__":
    train_yolo_classification()