import streamlit as st
import sys
import os

# Projenin kök dizinini (TheFishProject3) Python'un arama yoluna ekle
# Bu, 'src' klasörünü bulabilmesini sağlar
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.predict import predict

# --- Buradan sonraki kodunuz aynı kalabilir ---

st.set_page_config(layout="wide")
st.title("🦑 Kalamar Ağırlık Tahmin Uygulaması")

uploaded_file = st.file_uploader("Ağırlığını tahmin etmek için bir kalamar resmi yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Geçici dosya kaydetme
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Yüklenen Resim", use_column_width=True)

    with col2:
        with st.spinner('Ağırlık hesaplanıyor...'):
            prediction = predict(temp_path)
        st.success("Tahmin tamamlandı!")
        st.metric(label="Tahmini Ağırlık", value=f"{prediction:.2f} g")

    # Geçici dosyayı sil
    os.remove(temp_path)