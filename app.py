# -*- coding: utf-8 -*-
"""Klasifikasi Dampak Penggunaan Media Sosial terhadap Performa Akademik Mahasiswa dengan Decision Tree.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10Mbb1kTaq19xvB4SuU-eu1R8FG0FdO2L

# 1. Import Library
Mengimpor semua library yang diperlukan untuk manipulasi data (pandas, numpy), visualisasi (matplotlib, seaborn), preprocessing (LabelEncoder), model klasifikasi (DecisionTreeClassifier), dan evaluasi performa model.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

"""# 2. Load Dataset
Mengakses file CSV yang disimpan di Google Drive dan menampilkannya untuk melihat data secara sekilas
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

df = pd.read_csv('/content/drive/MyDrive/Machine Learning/Students Social Media Addiction.csv')
df.head()

"""# 3. Eksplorasi Awal Dataset
Memastikan tidak ada data kosong (null) dan bahwa tipe data sudah sesuai. Ini penting sebelum melanjutkan ke proses preprocessing atau pelatihan model
"""

# Cek kolom dan tipe data
df.info()

# Cek nilai null
print(df.isnull().sum())

"""Dari hasil pengecekan struktur data dengan df.info() dan df.isnull().sum(), dapat disimpulkan bahwa seluruh kolom dalam dataset telah memiliki tipe data yang sesuai dan tidak terdapat nilai yang hilang (null). Hal ini menunjukkan bahwa data dalam kondisi bersih dan siap digunakan untuk proses analisis lebih lanjut tanpa perlu melakukan imputasi atau pembersihan data tambahan, sehingga dapat meningkatkan efisiensi dalam pelatihan model machine learning

# 4. Encoding Kolom Kategorikal
Model machine learning hanya bisa bekerja dengan angka. Maka, semua kolom yang berisi data kategorikal (teks seperti "Male", "Instagram", dll) harus diubah menjadi nilai numerik dengan LabelEncoder
"""

data = df.copy()

# Identifikasi dan Encode Kolom Kategorikal
categorical_cols = ['Gender', 'Academic_Level', 'Country',
                    'Most_Used_Platform', 'Relationship_Status',
                    'Affects_Academic_Performance']

le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data.head()

"""Dari hasil encoding menggunakan LabelEncoder, seluruh kolom kategorikal dalam dataset—seperti Gender, Academic_Level, Country, Most_Used_Platform, Relationship_Status, dan Affects_Academic_Performance—telah berhasil dikonversi menjadi format numerik agar bisa digunakan oleh algoritma machine learning. Misalnya, Gender yang semula berisi nilai seperti "Male" dan "Female" kini diubah menjadi 0 dan 1, begitu juga dengan kategori lainnya. Hal ini penting karena model seperti Decision Tree hanya bisa memproses data numerik. Dataset hasil encoding ini siap digunakan untuk pelatihan model tanpa kehilangan makna dari nilai kategorikal aslinya

# 5. Memisahkan Fitur (X) dan Target (y)
X adalah fitur (variabel input), sedangkan y adalah target (output yang ingin diprediksi). Kita drop Student_ID karena tidak relevan, dan Affects_Academic_Performance dipisahkan sebagai label
"""

# Target: Affects_Academic_Performance (sudah di-encode)
X = data.drop(['Student_ID', 'Affects_Academic_Performance'], axis=1)
y = data['Affects_Academic_Performance']

"""# 6. Split Data: Training & Testing
Membagi data menjadi dua bagian:,

*   80% untuk pelatihan model (X_train, y_train)
*   20% untuk pengujian model (X_test, y_test)

Menggunakan random_state untuk hasil yang konsisten.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# 7. Latih Model Decision Tree
Membuat dan melatih model Decision Tree pada data pelatihan agar belajar pola dari fitur terhadap target
"""

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

"""# 8. Prediksi dan Evaluasi Model
Melakukan prediksi pada data uji, lalu mengevaluasi performa model dengan:
* Akurasi: seberapa tepat prediksi model
* Confusion Matrix: distribusi benar/salah prediksi
* Classification Report: precision, recall, dan f1-score
"""

y_pred = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""Hasil evaluasi model Decision Tree menunjukkan bahwa model mampu memprediksi dampak penggunaan media sosial terhadap performa akademik mahasiswa dengan akurasi sempurna, yaitu 100%. Confusion matrix memperlihatkan bahwa seluruh data uji, baik yang terdampak maupun tidak, berhasil diklasifikasikan dengan benar tanpa kesalahan. Nilai precision, recall, dan f1-score pada kedua kelas (0 dan 1) mencapai angka maksimal 1.00, yang mencerminkan performa klasifikasi yang sangat baik. Meskipun demikian, akurasi yang terlalu tinggi ini perlu dicermati lebih lanjut karena dapat mengindikasikan bahwa data sangat terstruktur atau model mengalami overfitting. Oleh karena itu, langkah selanjutnya adalah kami memeriksa kedalaman desicion tree-nya untuk memastikan bahwa model tetap sederhana dan tidak terlalu kompleks

# 9. Cek Kedalaman Decision Tree
Menampilkan seberapa dalam pohon keputusan yang terbentuk. Kedalaman yang rendah menunjukkan model yang sederhana namun bisa jadi sangat efektif
"""

print("Tree Depth:", model.get_depth())

"""Model Decision Tree yang dibangun menunjukkan akurasi sempurna sebesar 100% dalam mengklasifikasikan dampak penggunaan media sosial terhadap performa akademik mahasiswa, dan untuk memastikan hasil ini bukan disebabkan oleh overfitting, dilakukan pengecekan kedalaman pohon. Hasilnya menunjukkan bahwa kedalaman pohon hanya 2, yang berarti model cukup sederhana dan tidak kompleks. Dengan jumlah data sebanyak 705 baris, kedalaman yang dangkal ini menunjukkan bahwa model mampu menangkap pola yang jelas dalam data tanpa perlu banyak percabangan, sehingga hasil prediksi dapat dianggap akurat dan tidak berlebihan

# 10. Visualisasi Decision Tree
Menampilkan visual desicion tree, sehingga bisa melihat logika pengambilan keputusan model secara langsung.
"""

plt.figure(figsize=(12,6))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree untuk Prediksi Dampak Media Sosial")
plt.show()

"""Dalam proyek klasifikasi ini, kami membangun model Decision Tree untuk memprediksi dampak penggunaan media sosial terhadap performa akademik mahasiswa berdasarkan 705 baris data yang berisi berbagai atribut seperti jam tidur, tingkat konflik sosial media, dan platform yang digunakan. Model menunjukkan hasil evaluasi yang sangat baik dengan akurasi 100%, yang didukung oleh nilai precision, recall, dan f1-score yang sempurna. Untuk mengantisipasi kemungkinan overfitting, kami melakukan pengecekan kedalaman pohon keputusan, dan hasilnya hanya memiliki kedalaman 2, yang menunjukkan bahwa model sederhana namun efektif.

Visualisasi decision tree memperlihatkan bahwa dua fitur paling berpengaruh dalam prediksi adalah jumlah konflik akibat media sosial dan durasi tidur per malam. Pemisahan data berdasarkan fitur-fitur ini menghasilkan node yang sangat bersih (gini = 0), menandakan bahwa model berhasil membuat klasifikasi yang jelas dan logis. Dengan demikian, dapat disimpulkan bahwa variabel-variabel perilaku seperti konflik sosial media dan pola tidur memiliki pengaruh yang signifikan terhadap performa akademik, dan Decision Tree mampu menangkap pola ini secara efisien
"""