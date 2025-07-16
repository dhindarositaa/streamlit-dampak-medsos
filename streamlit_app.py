import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Klasifikasi Media Sosial vs Performa Akademik", layout="centered")

st.title("📊 Dampak Media Sosial terhadap Performa Akademik Mahasiswa")

# ======================
# 1. Load Dataset langsung dari file
# ======================
try:
    df = pd.read_csv("Students Social Media Addiction.csv")

    st.subheader("1. 📌 Data Awal")
    st.write(df.head())

    # Cek data null
    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Data mengandung nilai kosong. Mohon bersihkan dulu.")
    else:
        st.success("✅ Data bersih, tidak ada nilai kosong.")

        # Encode kategorikal
        data = df.copy()
        categorical_cols = ['Gender', 'Academic_Level', 'Country',
                            'Most_Used_Platform', 'Relationship_Status',
                            'Affects_Academic_Performance']

        le = LabelEncoder()
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])

        # Pisahkan fitur dan target
        X = data.drop(['Student_ID', 'Affects_Academic_Performance'], axis=1)
        y = data['Affects_Academic_Performance']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.subheader("2. 📈 Evaluasi Model")
        st.metric("Akurasi", f"{acc*100:.2f}%")
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]))
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        # Visualisasi Decision Tree
        st.subheader("3. 🌳 Visualisasi Decision Tree")

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
        st.pyplot(fig)

except FileNotFoundError:
    st.error("❌ File 'Students Social Media Addiction.csv' tidak ditemukan di folder.")
