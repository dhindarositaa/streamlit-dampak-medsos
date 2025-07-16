import streamlit as st
import pandas as pd
import joblib

# Load model dan encoder
model = joblib.load("model.pkl")

st.set_page_config(page_title="Prediksi Kecanduan Media Sosial", layout="centered")
st.title("📱 Prediksi Kecanduan Media Sosial Mahasiswa")

st.markdown("Masukkan data berikut untuk memprediksi apakah pengguna **kecanduan media sosial** berdasarkan model Decision Tree:")

# Form input
with st.form("predict_form"):
    umur = st.number_input("Umur", min_value=15, max_value=35, value=20)
    durasi = st.slider("Durasi Penggunaan Medsos per Hari (jam)", 0.0, 12.0, 2.0, step=0.1)
    tidur = st.slider("Jam Tidur per Malam", 0.0, 12.0, 6.0, step=0.5)
    
    submit = st.form_submit_button("Prediksi")

if submit:
    # Buat dataframe dari input user
    input_data = pd.DataFrame({
        "Age": [umur],
        "Duration": [durasi],
        "Sleep_Hours": [tidur]
    })

    # Kolom lain diisi default rata-rata (opsional, sesuaikan dengan training)
    # Jika kamu hanya melatih model dengan 3 fitur ini, langsung gunakan saja

    # Prediksi
    pred = model.predict(input_data)[0]
    label = "Kecanduan" if pred == 1 else "Tidak Kecanduan"

    st.subheader("📊 Hasil Prediksi:")
    st.success(f"Berdasarkan input yang diberikan, pengguna diprediksi: **{label}**")
