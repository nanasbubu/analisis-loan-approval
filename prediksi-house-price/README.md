# prediksi-house-price

Proyek ini mencakup alur analisis data harga rumah dari awal sampai akhir,
mulai dari pembersihan data hingga pemodelan.

Tahapan yang dikerjakan:
- Load dan pengecekan data (info, statistik deskriptif)
- Penanganan missing value (median, modus, dan drop kolom)
- Deteksi dan penyaringan outlier dengan metode IQR
- Standardisasi fitur numerik
- Penghapusan data duplikat
- Encoding fitur kategorikal (one-hot dan label encoding)
- Eksplorasi data (distribusi dan korelasi)
- Pembagian data latih dan uji
- Pelatihan model Lars, Linear Regression, dan Gradient Boosting Regressor
- Evaluasi model dengan MAE, MSE, dan R2
- Penyimpanan model dengan joblib
