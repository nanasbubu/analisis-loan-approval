# Klasifikasi dalam Machine Learning

## 1. Dasar Klasifikasi
Klasifikasi adalah tugas di machine learning di mana model memetakan sebuah data ke salah satu kelas yang sudah ditentukan sebelumnya. Klasifikasi termasuk supervised learning, yaitu pembelajaran dengan data latih yang memiliki pasangan input (fitur) dan output (label). Dari data tersebut model mempelajari hubungan antara fitur dan label, lalu menggunakannya untuk memprediksi kelas data baru yang labelnya belum diketahui. Dengan kata lain, klasifikasi adalah teknik untuk menentukan kelas dari sebuah observasi baru menggunakan data latih yang kelasnya sudah diketahui.

Hasil klasifikasi berupa kategori yang diskret, misalnya sehat atau sakit. Hal ini berbeda dengan regresi yang hasilnya berupa nilai numerik kontinu, seperti suhu atau harga. Karena jenis outputnya berbeda, metrik evaluasinya juga berbeda. Klasifikasi dievaluasi dengan akurasi, precision, recall, dan F1-score, sedangkan regresi dengan MSE, MAE, dan R2.

(sumber: Tim Pembina IOAI Indonesia, 2025, hlm. 1-2 dan 21; Molina Menéndez & Parraga-Alava, 2024, hlm. 1-2; Endut dkk., 2022, hlm. 659)

## 2. Jenis-Jenis Klasifikasi
Jenis klasifikasi dibedakan berdasarkan jumlah kelas dan jumlah label yang bisa dimiliki satu data, yaitu:
- Klasifikasi biner: memiliki dua kelas. Model memilih salah satu dari dua kemungkinan, misalnya memprediksi apakah seseorang menderita diabetes atau tidak.
- Klasifikasi multikelas: memiliki lebih dari dua kelas, dan setiap data hanya masuk ke satu kelas. Contohnya mengelompokkan objek menjadi orang, mobil, atau motor.
- Klasifikasi multilabel: memungkinkan satu data memiliki beberapa label sekaligus. Klasifikasi ini merupakan generalisasi dari multikelas karena tidak ada batas jumlah kelas yang bisa diberikan pada satu data. Aplikasinya antara lain kategorisasi teks, klasifikasi gambar, dan anotasi konten.

(sumber: Endut dkk., 2022, hlm. 658-659; Tim Pembina IOAI Indonesia, 2025, hlm. 2-4 dan 81; Molina Menéndez & Parraga-Alava, 2024, hlm. 1-2)

## 3. Algoritma yang Sering Digunakan
### 3.1 Logistic Regression

Logistic regression adalah algoritma klasifikasi, walaupun namanya mengandung kata regresi. Algoritma ini memprediksi probabilitas suatu data termasuk ke kelas tertentu, sehingga hasilnya bisa diubah menjadi prediksi kategori. Cara kerjanya, kombinasi linear dari fitur diubah menjadi nilai probabilitas antara 0 dan 1 menggunakan fungsi sigmoid. Pada contoh di modul, data diprediksi sebagai kelas 1 jika probabilitasnya minimal 0,5. Parameter modelnya dilatih dengan meminimalkan binary cross entropy menggunakan gradient descent. Modul tersebut menjelaskan logistic regression sebagai model klasifikasi biner, dengan contoh memprediksi apakah seorang siswa lulus ujian berdasarkan jam belajarnya.

(sumber: Tim Pembina IOAI Indonesia, 2025, hlm. 23-26; Endut dkk., 2022, hlm. 663)

### 3.2 K-Nearest Neighbors (KNN)
KNN mengklasifikasikan data baru dengan mencari k data latih yang jaraknya paling kecil, biasanya dengan jarak Euclidean, lalu memilih kelas yang paling banyak muncul di antara k data tersebut. KNN termasuk instance-based learning, yaitu algoritma yang memakai data latih secara langsung dan tidak membangun model terlebih dahulu.

Kelebihannya, KNN sederhana dan mudah diimplementasikan. Kekurangannya, waktu prediksi bisa lama pada dataset besar, hasilnya dipengaruhi skala fitur sehingga fitur perlu dinormalisasi, dan hasilnya sensitif terhadap nilai k serta outlier.

(sumber: Tim Pembina IOAI Indonesia, 2025, hlm. 27-29; Endut dkk., 2022, hlm. 663)

### 3.3 Support Vector Machine (SVM)
SVM mencari hyperplane yang memisahkan kelas dengan margin sebesar mungkin. Margin yang lebih lebar membuat model lebih baik dalam menggeneralisasi data baru. Data latih yang paling dekat dengan hyperplane disebut support vector. Prediksi kelas ditentukan dari tanda hasil fungsi keputusan linear f(x) = wTx + b.

Jika data mengandung noise, digunakan soft margin dengan parameter C yang mengatur keseimbangan antara lebar margin dan jumlah kesalahan klasifikasi. Jika data tidak bisa dipisahkan secara linear, digunakan fungsi kernel, misalnya linear, polynomial, RBF, atau sigmoid. SVM dapat menangani masalah berdimensi tinggi dan data yang tidak terpisah secara linear. Kelemahannya, kinerja SVM menurun jika data mengandung noise atau outlier. Pada tinjauan Endut dkk., SVM adalah algoritma yang paling sering digunakan untuk klasifikasi multilabel.

(sumber: Tim Pembina IOAI Indonesia, 2025, hlm. 29-32; Endut dkk., 2022, hlm. 659 dan 664; Ezugwu dkk., 2024, bagian 2.5)

### 3.4 Decision Tree

Decision tree menyusun keputusan dalam bentuk pohon. Node internal berisi pengujian terhadap sebuah fitur, cabang menunjukkan hasil pengujian, dan daun berisi kelas hasil prediksi. Pembentukan pohon dilakukan dari atas ke bawah, kemudian dilakukan pemangkasan (pruning) dari bawah ke atas. Pemisahan data dipilih berdasarkan tingkat kemurnian hasil pemisahan yang diukur dengan entropy atau Gini index. Semakin rendah nilainya, semakin baik pemisahannya.

Kelebihannya, decision tree mudah diinterpretasikan. Kekurangannya, pohon yang terlalu dalam mudah overfitting dan terlalu sensitif terhadap perubahan kecil pada data. Overfitting bisa dikurangi dengan pruning, pembatasan kedalaman maksimum, atau pembatasan jumlah data minimum pada daun. Ezugwu dkk. juga mencatat bahwa algoritma decision tree versi awal cenderung condong ke kelas dominan pada data yang tidak seimbang, tetapi kelemahan ini sudah banyak diperbaiki oleh pengembangan algoritma berikutnya.

(sumber: Tim Pembina IOAI Indonesia, 2025, hlm. 32-35 dan 37; Endut dkk., 2022, hlm. 663; Ezugwu dkk., 2024, bagian 2.1)

### 3.5 Random Forest
Random forest adalah metode ensemble yang menggabungkan banyak decision tree. Setiap pohon dilatih dari sampel acak dataset dengan pengambilan berulang (bootstrap), dan pada setiap pemisahan hanya sebagian fitur yang dipertimbangkan. Untuk klasifikasi, hasil akhirnya ditentukan dari voting mayoritas semua pohon.

Kelebihannya, random forest umumnya lebih akurat dan stabil daripada satu decision tree, mengurangi overfitting, tidak memerlukan normalisasi data, dan dapat menunjukkan fitur yang paling berpengaruh (feature importance). Kekurangannya, prosesnya lebih lambat, membutuhkan memori lebih besar, dan lebih sulit diinterpretasikan.

(sumber: Tim Pembina IOAI Indonesia, 2025, hlm. 34-35; Endut dkk., 2022, hlm. 663; Ezugwu dkk., 2024, bagian 2.2)

### 3.6 Naive Bayes
Naive Bayes adalah algoritma probabilistik yang berdasarkan Teorema Bayes. Algoritma ini mengasumsikan bahwa setiap fitur bersifat independen terhadap kelas. Asumsi tersebut umumnya lemah, tetapi dalam praktik Naive Bayes rutin dibandingkan dengan classifier yang lebih canggih. Menurut Endut dkk., pada kondisi ketika variabel input kategorikal cepat konvergen, Naive Bayes dapat bekerja lebih baik daripada model diskriminatif seperti logistic regression yang membutuhkan data latih lebih banyak. Naive Bayes dapat digunakan untuk klasifikasi biner maupun multikelas dan dapat bekerja dengan data latih yang terbatas.

Kekurangannya, asumsi independensi fitur tidak berlaku pada sebagian besar situasi nyata, kinerjanya kurang baik pada data yang tidak seimbang, dan muncul masalah probabilitas nol jika suatu nilai fitur pada data uji untuk kelas tertentu tidak pernah ditemukan pada data latih.

(sumber: Endut dkk., 2022, hlm. 659 dan 664; Ezugwu dkk., 2024, bagian 2.3)

### 3.7 Neural Networks
Neural network atau jaringan saraf tiruan tersusun dari elemen pemroses yang saling terhubung, disebut neuron. Jaringan ini digunakan antara lain untuk pengenalan pola, data mining, dan diagnosis medis. Bentuk paling awalnya adalah perceptron yang diperkenalkan pada tahun 1950-an. Perceptron tidak dapat mengklasifikasikan pola yang tidak terpisah secara linear dan hanya menghasilkan keluaran biner, sehingga tidak cocok untuk klasifikasi multikelas. Keterbatasan ini diatasi dengan multilayer perceptron (MLP), yang terdiri dari input layer, satu atau lebih hidden layer, dan output layer, serta dilatih dengan backpropagation.

Pada tinjauan terhadap 30 studi tahun 2019-2024, convolutional neural network (CNN) paling banyak digunakan (14 dari 30 studi), terutama untuk data gambar, diikuti MLP untuk klasifikasi secara umum. Pemilihan arsitektur bergantung pada karakteristik data, dan kinerja jaringan dipengaruhi hyperparameter seperti jumlah node pada hidden layer dan learning rate.

(sumber: Ezugwu dkk., 2024, bagian 2.8; Molina Menéndez & Parraga-Alava, 2024, hlm. 1-2 dan bagian Results (RQ1))

## 4. Evaluasi Model Klasifikasi
Kinerja model klasifikasi dinilai menggunakan data uji yang tidak dipakai saat pelatihan. Dataset umumnya dibagi menjadi data latih untuk melatih model, data validasi untuk menyetel hyperparameter dan mencegah overfitting, dan data uji untuk evaluasi akhir.

| Metrik | Rumus | Keterangan |
|---|---|---|
| Akurasi | prediksi benar / total data | Persentase prediksi yang benar |
| Precision | TP / (TP + FP) | Proporsi prediksi positif yang benar |
| Recall | TP / (TP + FN) | Proporsi data positif yang berhasil ditemukan |
| F1-score | 2 × (Precision × Recall) / (Precision + Recall) | Rata-rata harmonis precision dan recall |

Akurasi bisa menyesatkan pada data yang tidak seimbang. Misalnya, jika 95% data berasal dari satu kelas, model yang selalu memprediksi kelas tersebut akan mendapat akurasi 95% walaupun tidak berguna. Karena itu, precision, recall, dan F1-score perlu digunakan bersama akurasi.

(sumber: Tim Pembina IOAI Indonesia, 2025, hlm. 21-23, 37, dan 73; Hakim dkk., 2024, hlm. 15-17)

## 5. Kesimpulan
Klasifikasi adalah tugas supervised learning yang jenisnya dibedakan menjadi biner, multikelas, dan multilabel. Setiap algoritma memiliki kelebihan dan kekurangan, dan pemilihannya bergantung pada karakteristik data yang digunakan.

## Daftar Pustaka
- Endut, N., Hamzah, W. M. A. F. W., Ismail, I., Yusof, M. K., Abu Baker, Y., & Yusoff, H. (2022). A systematic literature review on multi-label classification based on machine learning algorithms. TEM Journal, 11(2), 658-666. https://doi.org/10.18421/TEM112-20
- Ezugwu, A. E., Ho, Y.-S., Egwuche, O. S., Ekundayo, O. S., Van Der Merwe, A., Saha, A. K., & Pal, J. (2024). Classical machine learning: Seventy years of algorithmic learning evolution. Data Intelligence (MIT Press), accepted for publication. https://arxiv.org/abs/2408.01747
- Hakim, L., Sobri, A., Sunardi, L., & Nurdiansyah, D. (2024). Prediksi penyakit jantung berbasis mesin learning dengan menggunakan metode k-nn. Jurnal Digital Teknologi Informasi, 7(2), 14-20. https://doi.org/10.32502/digital.v7i2.9429
- Molina Menéndez, E., & Parraga-Alava, J. (2024). Artificial neural networks for classification tasks: A systematic literature review. Enfoque UTE, 15(4), 1-10. https://doi.org/10.29019/enfoqueute.1058
- Tim Pembina IOAI Indonesia. (2025). Modul Pelatnas IOAI Indonesia, Modul 2: Pengenalan Machine Learning. https://ioai.toki.id/assets/modul/Modul_2_IOAI__Pengenalan_Machine_Learning__ML_.pdf
