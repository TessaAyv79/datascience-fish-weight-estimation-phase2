import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast # Metin halindeki listeleri ayrıştırmak için

# Çıktı klasörlerini oluştur
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Görselleştirme için stil ayarları
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

def perform_comprehensive_eda(csv_path='data/squid_dataset.csv'):
    """Veri setini yükler, detaylı analiz eder, görselleştirir ve temizlenmiş halini kaydeder."""
    
    print("--- 🔬 KAPSAMLI VERİ ANALİZİ BAŞLADI 🔬 ---")

    # === 1. VERİYİ YÜKLEME VE İLK BAKIŞ ===
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✅ Veri Seti Başarıyla Yüklendi: {csv_path}")
        print(f"Veri Seti Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
    except FileNotFoundError:
        print(f"❌ HATA: {csv_path} dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return

    print("\n--- Veri Seti İlk 5 Satır ---")
    print(df.head())

    print("\n--- Veri Tipleri ve Bellek Kullanımı ---")
    df.info()

    # === 2. İSTATİSTİKSEL ÖZET VE EKSİK VERİ ===
    print("\n--- Sayısal Sütunların İstatistiksel Özeti ---")
    print(df.describe().T)
    
    print("\n--- Eksik Veri Raporu ---")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({'Eksik Sayısı': missing_values, 'Yüzde (%)': missing_percent})
    missing_df = missing_df[missing_df['Eksik Sayısı'] > 0].sort_values(by='Yüzde (%)', ascending=False)
    
    if not missing_df.empty:
        print("❗️ DİKKAT: Aşağıdaki sütunlarda eksik veri bulundu:")
        print(missing_df)
    else:
        print("✅ Tüm sütunlar dolu, eksik veri bulunmuyor.")

    # === 3. VERİ TEMİZLİĞİ VE DÖNÜŞÜM ===
    print("\n--- Veri Temizliği ve Dönüşüm ---")
    
    # 'Defect' sütununu güvenli bir şekilde ayrıştır
    df['Defect_List'] = df['Defect'].apply(lambda s: ast.literal_eval(s) if pd.notna(s) and isinstance(s, str) and s.startswith('[') else [])
    
    # Sayısal sütunlarda 0 veya negatif olabilecek anlamsız değerleri kontrol et
    print("Ağırlık <= 0 olan satır sayısı:", (df['Total Weight (g)'] <= 0).sum())
    print("Uzunluk <= 0 olan satır sayısı:", (df['Total Length (cm)'] <= 0).sum())
    
    # Temizlenmiş DataFrame oluştur
    df_cleaned = df[
        (df['Total Weight (g)'] > 0) & 
        (df['Total Length (cm)'] > 0)
    ].copy()
    print(f"Anlamsız değerler temizlendi. Kalan satır sayısı: {len(df_cleaned)}")

    # === 4. GÖRSEL VERİ ANALİZİ (EDA) ===
    print("\n--- Görsel Analizler Oluşturuluyor ve Kaydediliyor ---")

    # Sayısal Değişkenlerin Dağılımı ve Aykırı Değerler
    for col in ['Total Weight (g)', 'Total Length (cm)']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(df_cleaned[col], kde=True, ax=axes[0], bins=50)
        axes[0].set_title(f'{col} Dağılımı (Histogram)')
        sns.boxplot(x=df_cleaned[col], ax=axes[1])
        axes[1].set_title(f'{col} Dağılımı (Kutu Grafiği - Aykırı Değerler)')
        plt.tight_layout()
        filepath = os.path.join(FIGURES_DIR, f'{col.replace(" ", "_").lower()}_distribution.png')
        plt.savefig(filepath)
        print(f"✅ Grafik kaydedildi: {filepath}")
        plt.close()

    # Kategorik Değişkenlerin Dağılımı
    for col in ['Color', 'Species']:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df_cleaned[col], order=df_cleaned[col].value_counts().index, palette='viridis')
        plt.title(f'{col} Sütunu Değer Sayıları')
        plt.xlabel('Sayı')
        plt.ylabel(col)
        plt.tight_layout()
        filepath = os.path.join(FIGURES_DIR, f'{col.lower()}_counts.png')
        plt.savefig(filepath)
        print(f"✅ Grafik kaydedildi: {filepath}")
        plt.close()

    # Uzunluk ve Ağırlık İlişkisi
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_cleaned, x='Total Length (cm)', y='Total Weight (g)', hue='Species', alpha=0.7)
    plt.title('Kalamar Uzunluğu ve Ağırlığı Arasındaki İlişki (Türe Göre)')
    filepath = os.path.join(FIGURES_DIR, 'length_vs_weight_scatter.png')
    plt.savefig(filepath)
    print(f"✅ Grafik kaydedildi: {filepath}")
    plt.close()

    # Korelasyon Isı Haritası
    plt.figure(figsize=(8, 6))
    numeric_cols = df_cleaned.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Sayısal Değişkenler Arası Korelasyon')
    filepath = os.path.join(FIGURES_DIR, 'correlation_heatmap.png')
    plt.savefig(filepath)
    print(f"✅ Grafik kaydedildi: {filepath}")
    plt.close()

    # === 5. TEMİZLENMİŞ VERİYİ KAYDETME ===
    cleaned_csv_path = 'data/cleaned_squid_dataset.csv'
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    print(f"\n✅ Temizlenmiş veri seti şuraya kaydedildi: {cleaned_csv_path}")

    print("\n--- ✨ Analiz Başarıyla Tamamlandı! ✨ ---")

if __name__ == '__main__':
    perform_comprehensive_eda()