# Analisis Loan Approval

Proyek klasifikasi untuk memprediksi persetujuan pinjaman (`loan_status`) menggunakan dataset Loan Approval dari Kaggle. Dibuat untuk tugas mata kuliah [nama mata kuliah] oleh [nama kamu].

## Dataset

- Sumber: [Loan Approval Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- Jumlah data: 45.000 baris dan 14 kolom
- Target: `loan_status` (1 = disetujui, 0 = ditolak)
- Fitur: usia, jenis kelamin, pendidikan, income, status kepemilikan rumah, jumlah dan tujuan pinjaman, suku bunga, rasio pinjaman terhadap income, lama riwayat kredit, credit score, dan riwayat gagal bayar sebelumnya

## Tahapan

1. Data loading
2. Data cleaning dan transformation: cek missing value dan duplikat, hapus usia tidak valid (di atas 100 tahun), hapus kolom `person_emp_exp`, tangani outlier dengan capping (IQR), one-hot encoding
3. Exploratory Data Analysis (EDA): distribusi, hubungan fitur dengan target, dan korelasi
4. Data splitting (80:20 dengan stratify) dan standardisasi fitur numerik
5. Penanganan data tidak seimbang dengan SMOTE (hanya di data latih)
6. Modelling dan evaluasi: Logistic Regression, Random Forest, Gradient Boosting, dan XGBoost

## Hasil

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0,862 | 0,634 | 0,903 | 0,745 | 0,950 |
| Random Forest | 0,893 | 0,699 | 0,909 | 0,790 | 0,970 |
| Gradient Boosting | 0,891 | 0,696 | 0,901 | 0,785 | 0,969 |
| XGBoost | 0,918 | 0,789 | 0,860 | 0,823 | 0,973 |

XGBoost memberikan hasil terbaik di hampir semua metrik. Fitur yang paling berpengaruh adalah riwayat gagal bayar sebelumnya (`previous_loan_defaults_on_file`).

## Struktur File

- `analisis_loan_approval.ipynb`: notebook utama berisi seluruh tahapan analisis
- `loan_data.csv`: dataset
- `loan_model.joblib`: model XGBoost yang sudah dilatih (hapus baris ini kalau file-nya tidak kamu upload)

## Cara Menjalankan

1. Buka `analisis_loan_approval.ipynb` di Google Colab
2. Jalankan semua sel secara berurutan (Runtime, lalu Run all). Dataset dibaca langsung dari repo ini
3. Pustaka yang dipakai: pandas, numpy, seaborn, matplotlib, scikit-learn, imbalanced-learn, dan xgboost

## Catatan

Angka hasil bisa sedikit berbeda jika `random_state` atau parameter model diubah.
