import streamlit as st
import pandas as pd
import joblib
import os

# 1. Pengaturan Halaman Utama
st.set_page_config(page_title="Prediksi Churn E-Commerce", page_icon="🎯", layout="wide")

# 2. Fungsi Memuat Model (Dengan Pelacakan Direktori Absolut)
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'churn_prediction_model.pkl')
    return joblib.load(model_path)

# 3. INISIALISASI MODEL (Ini adalah baris yang hilang sebelumnya)
model = load_model()

# 4. Antarmuka Pengguna (UI) - Judul
st.title("🎯 Sistem Prediksi Risiko Churn Pelanggan")
st.markdown("Masukkan data operasional pelanggan di bawah ini untuk mengevaluasi probabilitas retensi secara *real-time*.")
st.divider()

# 5. Formulir Input Pengguna (Dibagi 2 Kolom)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Transaksi & Loyalitas")
    tenure = st.number_input("Masa Berlangganan (Tenure dalam Bulan)", min_value=0, max_value=60, value=1)
    cashback = st.number_input("Rata-rata Cashback (Rupee)", min_value=0.0, value=150.0)
    day_since_order = st.number_input("Hari Sejak Pesanan Terakhir", min_value=0, value=10)

with col2:
    st.subheader("Data Pengalaman Pelanggan")
    satisfaction = st.slider("Skor Kepuasan (1-5)", min_value=1, max_value=5, value=3)
    complain = st.selectbox("Riwayat Komplain", options=["Tidak Ada (0)", "Ada (1)"])
    
    # Ekstrak nilai numerik dari pilihan komplain
    complain_val = 1 if complain == "Ada (1)" else 0

st.divider()

# 6. Logika Tombol Prediksi
if st.button("Jalankan Prediksi Risiko", type="primary", use_container_width=True):
    
    # MENYUSUN DATA INPUT
    # PENTING: Nama kolom di bawah ini HARUS sama persis (huruf besar/kecilnya) 
    # dengan dataset yang Anda gunakan saat melatih model Random Forest.
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'CashbackAmount': [cashback],
        'DaySinceLastOrder': [day_since_order],
        'SatisfactionScore': [satisfaction],
        'Complain': [complain_val]
    })

    # Mengeksekusi Prediksi
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        # Menampilkan Hasil
        st.subheader("Hasil Analisis Sistem:")
        if prediction == 1:
            st.error(f"⚠️ PELANGGAN BERISIKO TINGGI (CHURN) | Probabilitas: {probability:.1f}%")
            st.info("Saran Tindakan: Segera kirimkan kampanye retensi atau hubungi pelanggan secara personal.")
        else:
            st.success(f"✅ PELANGGAN AMAN (LOYAL) | Probabilitas Churn: {probability:.1f}%")
            st.info("Saran Tindakan: Pertahankan kualitas layanan saat ini.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data input: {e}")
        st.warning("Pastikan jumlah dan nama variabel input di atas sudah persis sama dengan variabel yang digunakan saat melatih model (X_train).")
