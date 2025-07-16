import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Skor Kecanduan Medsos", layout="centered")
st.title("📱 Prediksi Skor Kecanduan Media Sosial Mahasiswa")

@st.cache_data
def train_model():
    df = pd.read_csv("Students Social Media Addiction.csv")

    # Pastikan nama kolom sesuai dengan file asli
    X = df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
    y = df['Addicted_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

st.subheader("📝 Masukkan Data Anda")

with st.form("form_prediksi"):
    umur = st.number_input("Umur", min_value=15, max_value=35, value=20)
    durasi = st.slider("Durasi Penggunaan Medsos (jam/hari)", 0.0, 12.0, 3.0, step=0.1)
    tidur = st.slider("Jam Tidur (per malam)", 0.0, 12.0, 6.0, step=0.5)
    submit = st.form_submit_button("Prediksi Skor")

if submit:
    input_df = pd.DataFrame({
        'Age': [umur],
        'Avg_Daily_Usage_Hours': [durasi],
        'Sleep_Hours_Per_Night': [tidur]
    })

    pred_score = int(model.predict(input_df)[0])
    status = "⚠️ Kecanduan" if pred_score >= 6 else "✅ Tidak Kecanduan"

    st.subheader("📊 Hasil Prediksi")
    st.info(f"Skor Kecanduan: **{pred_score}**")
    st.success(f"Status: **{status}**")
