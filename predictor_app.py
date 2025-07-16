import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ====================
# 1. Load & Preprocess Data
# ====================
@st.cache_data
def load_model():
    df = pd.read_csv("Students Social Media Addiction.csv")

    categorical_cols = ['Gender', 'Academic_Level', 'Country',
                        'Most_Used_Platform', 'Relationship_Status',
                        'Affects_Academic_Performance']
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df[['Age', 'Duration', 'Sleep_Hours']]
    y = df['Affects_Academic_Performance']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_model()

# ====================
# 2. UI Streamlit
# ====================
st.set_page_config(page_title="Prediksi Kecanduan Medsos", layout="centered")
st.title("📱 Prediksi Kecanduan Media Sosial Mahasiswa")

with st.form("form_predict"):
    umur = st.number_input("Umur", min_value=15, max_value=35, value=20)
    durasi = st.slider("Durasi Penggunaan Medsos (jam/hari)", 0.0, 12.0, 3.0, step=0.1)
    tidur = st.slider("Jam Tidur (per malam)", 0.0, 12.0, 6.0, step=0.5)
    submit = st.form_submit_button("Prediksi")

if submit:
    input_data = pd.DataFrame({
        'Age': [umur],
        'Duration': [durasi],
        'Sleep_Hours': [tidur]
    })

    pred = model.predict(input_data)[0]
    hasil = "✅ Tidak Kecanduan" if pred == 0 else "⚠️ Kecanduan"

    st.subheader("📊 Hasil Prediksi:")
    st.success(f"Hasil prediksi: **{hasil}**")
