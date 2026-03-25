import streamlit as st
import pandas as pd
import joblib

# 1. Pengaturan Halaman
st.set_page_config(page_title="Prediksi Churn E-Commerce", page_icon="🎯", layout="wide")

# 2. Memuat Model
@st.cache_resource
def load_model():
    return joblib.load('churn_prediction_model.pkl') 

model = load_model()

# 3. Judul dan Deskripsi
st.title("🎯 Sistem Prediksi Risiko Churn Pelanggan")
st.markdown("Masukkan data operasional pelanggan di bawah ini untuk mengevaluasi probabilitas retensi secara *real-time*.")
st.divider()

# 4. Formulir Input Pengguna (Dibagi 2 Kolom)
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

# 5. Tombol Prediksi
st.divider()
if st.button("Jalankan Prediksi Risiko", type="primary", use_container_width=True):
    # Menyusun data input menjadi DataFrame
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'CashbackAmount': [cashback],
        'DaySinceLastOrder': [day_since_order],
        'SatisfactionScore': [satisfaction],
        'Complain': [complain_val]
        # Tambahkan variabel lain yang dibutuhkan model Anda di sini
    })

    # Melakukan prediksi
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    # Menampilkan Hasil
    st.subheader("Hasil Analisis Sistem:")
    if prediction == 1:
        st.error(f"⚠️ PELANGGAN BERISIKO TINGGI (CHURN) | Probabilitas: {probability:.1f}%")
        st.info("Saran Tindakan: Segera kirimkan kampanye retensi atau hubungi pelanggan.")
    else:
        st.success(f"✅ PELANGGAN AMAN (LOYAL) | Probabilitas Churn: {probability:.1f}%")
        st.info("Saran Tindakan: Pertahankan kualitas layanan saat ini.")
