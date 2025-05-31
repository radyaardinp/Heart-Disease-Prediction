# Laporan Proyek Machine Learning - Radya Ardi Ninang Pudyastuti
## Domain Proyek
  Penyakit jantung merupakan penyebab utama kematian secara global. Berdasarkan data World Health Organization (WHO), sekitar 17,9 juta orang meninggal setiap tahunnya akibat penyakit kardiovaskular, termasuk serangan jantung dan stroke [1]. Oleh karena itu, deteksi dini terhadap penyakit jantung sangat penting agar intervensi medis dapat dilakukan lebih awal dan tepat sasaran.Seiring berkembangnya teknologi dan ketersediaan data medis, pendekatan Machine Learning (ML) telah menjadi alat bantu yang menjanjikan dalam bidang kesehatan, khususnya untuk prediksi penyakit. Dengan memanfaatkan data historis pasien, Machine Learning mampu membangun model klasifikasi yang dapat memprediksi kemungkinan seseorang mengalami penyakit jantung. 
 
  Beberapa penelitian telah menunjukkan bahwa efektifitas metode Machine Learning mampu memprediksi penyakit jantung. Misalnya, penelitian yang dilakukan oleh Kwandy dan Juan Sebastian (2024), metode Machine Learning Logistic Regression dan K-Nearest Neighbour digunakan untuk memprediksi penyakit jantung koroner. Dari penelitian tersebut, didapatkan model Regresi Logistik memiliki kinerja yang lebih baik dalam memprediksi penyakit jantung koroner dibandingkan K-NN, dengan akurasi 91,8%, presisi 88%, recall 92%, dan F1-score 89%. Sedangkan model K-NN menunjukkan akurasi 91,5%, presisi 88%, recall 92%, dan F1-score 89% [2]. Penelitian lain dilakukan oleh Hidayat et al. (2024), mengimplementasikan metode Support Vector Machine (SVM) untuk prediksi penyakit jantung. Penelitian ini mendapatkan nilai  accuracy sebesar 0,85,precision sebesar 0,93, recall sebesar 0,76, dan f-1 score sebesar 0,83 [3]. 

  Dari penelitian di atas, Penelitian ini menghadirkan kebaruan melalui penggunaan dataset dari platform Kaggle yang berjudul _Heart Disease Prediction Dataset_, yang berisi data medis terstruktur untuk mendukung proses klasifikasi penyakit jantung. Dalam proses analisis, digunakan tiga algoritma machine learning, yaitu Logistic Regression (LR), Support Vector Machine (SVM), dan K-Nearest Neighbors (KNN), yang telah terbukti memiliki performa baik dalam berbagai penelitian terdahulu. Setelah model dilatih dan dievaluasi, algoritma dengan hasil terbaik akan dipilih untuk menjalani proses hyperparameter tuning guna memaksimalkan akurasi prediksi. Dengan langkah ini, diharapkan model yang dikembangkan dapat menghasilkan prediksi yang akurat dan dapat diandalkan, khususnya dalam mendukung pengambilan keputusan di bidang medis untuk deteksi dini penyakit jantung.

referensi:


## Business Understanding
### Problem Statements
1. Bagaimana cara mengklasifikasikan apakah seseorang berisiko terkena penyakit jantung atau tidak berdasarkan data medis yang tersedia?
2. Algoritma machine learning apa yang paling efektif dalam memprediksi penyakit jantung pada dataset yang digunakan?
3. Bagaimana cara meningkatkan performa model prediksi agar akurasi klasifikasi menjadi lebih optimal?

### Goals
1. Mengembangkan model klasifikasi yang mampu mengidentifikasi risiko penyakit jantung berdasarkan fitur-fitur medis pasien.
2. Membandingkan performa tiga algoritma machine learning, yaitu Logistic Regression (LR), Support Vector Machine (SVM), dan K-Nearest Neighbors (KNN) dalam memprediksi penyakit jantung.
3. Melakukan optimasi model terbaik menggunakan hyperparameter tuning untuk memperoleh hasil prediksi yang lebih akurat.

### Solution Statements
1. Mengimplementasikan dan membandingkan tiga algoritma machine learning, yaitu Logistic Regression (LR), Support Vector Machine (SVM), dan K-Nearest Neighbors (KNN), menggunakan dataset dari Kaggle.
2. Melakukan proses evaluasi model menggunakan metrik seperti akurasi, precision, recall, dan F1-score untuk mengetahui algoritma dengan performa terbaik.
3. Menerapkan teknik hyperparameter tuning (misalnya GridSearchCV) pada algoritma terbaik untuk meningkatkan kinerja model secara signifikan.

## Data Understanding
Dataset yang digunakan dalam penelitian ini adalah [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data) dari platform kaggle. Dataset ini berisi 1000 data rekam medis pasien yang dengan 16 atribut klinis dan kebiasaan gaya hidup yang berkaitan dengan risiko penyakit jantung. Data ini digunakan untuk melakukan klasifikasi biner, yaitu memprediksi apakah seorang individu memiliki penyakit jantung (1) atau tidak (0), yang ditunjukkan oleh kolom Heart Disease.

Adapun penjelasan masing-masing variabel pada dataset adalah sebagai berikut:
### Variabel Pada Heart Disease Dataset

| Nama Kolom                | Deskripsi                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| `Age`                     | Usia pasien (dalam tahun)                                                                         |
| `Gender`                  | Jenis kelamin pasien (Male / Female)                                                              |
| `Cholesterol`             | Kadar kolesterol pasien (dalam mg/dL)                                                             |
| `Blood Pressure`          | Tekanan darah pasien saat istirahat (dalam mm Hg)                                                 |
| `Heart Rate`              | Detak jantung pasien (denyut per menit)                                                           |
| `Smoking`                 | Status merokok pasien (Never, Current, Former)                                                    |
| `Alcohol Intake`          | Frekuensi konsumsi alkohol (Heavy, Moderate, atau kosong/NaN)                                     |
| `Exercise Hours`          | Jumlah jam olahraga per minggu                                                                    |
| `Family History`          | Riwayat keluarga dengan penyakit jantung (Yes / No)                                               |
| `Diabetes`                | Apakah pasien memiliki diabetes (Yes / No)                                                        |
| `Obesity`                 | Apakah pasien mengalami obesitas (Yes / No)                                                       |
| `Stress Level`            | Tingkat stres pasien (skala numerik, kemungkinan 1–10)                                            |
| `Blood Sugar`             | Kadar gula darah pasien (mg/dL)                                                                   |
| `Exercise Induced Angina` | Apakah pasien mengalami angina akibat olahraga (Yes / No)                                         |
| `Chest Pain Type`         | Jenis nyeri dada yang dirasakan (Atypical Angina, Typical Angina, Non-anginal Pain, Asymptomatic) |
| `Heart Disease`           | Label target: 1 jika pasien memiliki penyakit jantung, 0 jika tidak                               |

### Insight yang di dapat dari data 
- Distribusi label target tidak seimbang, yang artinya adalah Jumlah pasien yang tidak mengalami penyakit jantung memiliki rasio lebih banyak dari pada pasien yang memiliki penyakit jantung. 

   ![image](https://github.com/user-attachments/assets/03a0b82b-6270-436d-a22a-9514cdab545e)
- Usia pasien berkisar antara usia muda hingga lanjut usia, dengan frekuensi tertinggi ada di umur >35 tahun.

   ![image](https://github.com/user-attachments/assets/6d2daa59-3ff4-459d-a405-770c57fd63ab)

## Data Preparation
Data Preparation
Pada tahap ini, dilakukan serangkaian proses untuk menyiapkan data sebelum masuk ke proses pelatihan model. Langkah-langkah yang dilakukan disusun secara sistematis untuk memastikan data bersih, konsisten, dan relevan dengan tujuan klasifikasi. Berikut adalah tahapan data preparation yang dilakukan:

**1. Menghapus Duplikasi**
Langkah awal dilakukan pengecekan dan penghapusan data duplikat agar tidak memengaruhi hasil pelatihan model. Data duplikat dapat menyebabkan bias dan overfitting pada algoritma.

**2. Menangani Data Kosong (Missing Values)**
Ditemukan beberapa nilai kosong (NaN) pada kolom Alcohol Intake. Oleh karena itu nilai kosong tersebut dilakukan tahapan imputasi data diisi menggunakan mean (nilai rata-rata).

**3. Encoding Variabel Kategorikal**
Beberapa fitur memiliki tipe data kategorikal yang tidak dapat langsung digunakan oleh model machine learning menggunakan teknik Label encoding 

**4. Feature Scaling**
Agar model dapat belajar secara optimal, fitur numerik seperti Cholesterol, Blood Pressure, Heart Rate, Exercise Hours, Stress Level, dan Blood Sugar dinormalisasi menggunakan Min-Max Scaling ke rentang 0–1. Ini bertujuan untuk menyamakan skala antar fitur sehingga tidak mendominasi satu sama lain saat model dilatih.
![image](https://github.com/user-attachments/assets/33f2b8fe-1682-4837-9dc9-50f416d71ddd)

**5. Pemisahan Fitur dan Label**
Data dipisahkan menjadi:
Fitur (X): Seluruh kolom kecuali Heart Disease
Label (y): Kolom Heart Disease yang menjadi target klasifikasi

**6. Split Data**
Data dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan train_test_split. 

# Modeling
Dalam proyek ini, tiga algoritma machine learning digunakan untuk memodelkan prediksi penyakit jantung, yaitu Logistic Regression (LR), Support Vector Machine (SVM), dan K-Nearest Neighbors (KNN). Data dibagi menjadi dua bagian menggunakan teknik train-test split dengan rasio 80:20, di mana 80% data digunakan untuk pelatihan dan 20% sisanya untuk pengujian.

**1. Logistic Regression (LR)**
Logistic Regression adalah algoritma klasifikasi yang sederhana dan interpretatif, cocok untuk memodelkan hubungan antara satu atau lebih variabel independen dengan variabel dependen kategorikal.

Parameter yang digunakan:
- random_state=42: untuk menjaga hasil tetap konsisten.
- max_iter=1000: meningkatkan jumlah iterasi agar model dapat konvergen.

**2. Support Vector Machine (SVM)**
SVM bertujuan mencari hyperplane optimal yang memisahkan kelas-kelas dalam data. Metode ini efektif dalam menangani data berdimensi tinggi dan non-linear (terutama dengan kernel RBF).

Parameter yang digunakan:
random_state=42: agar hasil konsisten.

**3. K-Nearest Neighbors (KNN)**
KNN mengklasifikasikan data berdasarkan mayoritas kelas dari k tetangga terdekat (berdasarkan jarak). Algoritma ini mudah dipahami dan diimplementasikan, namun sensitif terhadap skala fitur dan pemilihan nilai k.

Parameter yang digunakan:
n_neighbors=5: jumlah tetangga yang dipertimbangkan.

Setelah ketiga model dilatih dan diuji, didapatkan performa awal sebagai berikut:
| Model               | Akurasi |
| ------------------- | ------- |
| Logistic Regression | 0.860   |
| SVM (Awal)          | 0.930   |
| KNN                 | 0.875   |

### Hyperparameter Tuning
Berdasarkan hasil evaluasi awal, model SVM menunjukkan akurasi terbaik. Oleh karena itu, dilakukan proses optimalisasi model ini menggunakan teknik GridSearchCV untuk mencari kombinasi parameter terbaik melalui validasi silang.

Parameter Grid yang Digunakan:

``param_grid = {
C: [0.1, 1, 10, 100],         # Parameter regularisasi``

``gamma: [1, 0.1, 0.01, 0.001], # Parameter untuk kernel RBF``

``kernel: ['rbf', 'linear']     # Jenis kernel``


Hasil Tuning:
Best Parameters: C=10, gamma=0.001, kernel='rbf'

Best Cross-validation Accuracy: 0.9275

## Evaluasi
Untuk mengevaluasi performa model dalam memprediksi penyakit jantung, digunakan beberapa metrik evaluasi berikut:

**1. Accuracy**: Proporsi prediksi yang benar dari seluruh prediksi yang dilakukan oleh model.
![image](https://github.com/user-attachments/assets/fc9cf6bd-eafc-4288-a8e0-3527efa851a5)

**2. Precision**: Proporsi prediksi positif yang benar-benar positif.
![image](https://github.com/user-attachments/assets/d1ee87aa-00bb-4e85-a94e-0ee92794d2dc)

**3. Recall** : Proporsi kasus positif yang berhasil dideteksi oleh model.
![image](https://github.com/user-attachments/assets/d4bd7a6c-2138-4901-bf86-69ddb1842780)

**4. F1-Score** : Rata-rata harmonik dari precision dan recall.
![image](https://github.com/user-attachments/assets/80900861-6351-4824-9f84-4b7ee1e2497d)

Keterangan:
- TP (True Positive): Model memprediksi positif dan memang positif
- TN (True Negative): Model memprediksi negatif dan memang negatif
- FP (False Positive): Model memprediksi positif, sebenarnya negatif
- FN (False Negative): Model memprediksi negatif, sebenarnya positif

### Hasil Evaluasi Model
|             Model            |  Accuracy  |	Precision	| Recall	| F1-Score|
|------------------------------|------------|-----------|---------|---------|
|Logistic Regression           |    0.86    |    0.84 	|   0.82	|   0.83  |
|K-Nearest Neighbors           |	  0.875   |    0.85   |   0.84  |	  0.85  |
|Support Vector Machine (SVM)  |	  0.93    |    0.96	  |   0.87  |   0.91  |
|SVM (Optimized)               |  	0.92    |    0.90   |   0.90  |   0.90  |

### Kesimpulan
Berdasarkan hasil evaluasi model menggunakan metrik akurasi, presisi, recall, dan F1-score, diperoleh bahwa:
- Model SVM awal memberikan akurasi tertinggi sebesar 93%, dengan presisi dan recall yang cukup seimbang untuk kedua kelas.
- Setelah dilakukan hyperparameter tuning menggunakan GridSearchCV, akurasi menurun sedikit menjadi 92%, namun model menjadi lebih seimbang dalam mendeteksi kedua kelas, dengan precision dan recall untuk kelas 1 meningkat dibanding Logistic Regression dan KNN.
- Model Logistic Regression menunjukkan performa yang cukup baik dengan akurasi 86%, namun kurang optimal dalam mendeteksi kelas positif.
- Model KNN memiliki akurasi 87.5% dengan performa yang sedikit lebih baik dari Logistic Regression, namun masih berada di bawah SVM.

Secara keseluruhan, SVM menjadi model terbaik untuk kasus ini karena memberikan performa yang paling stabil dan tinggi di semua metrik evaluasi utama, serta tetap seimbang setelah dilakukan proses tuning. Metrik yang digunakan juga sesuai dengan konteks data karena deteksi penyakit jantung memerlukan keseimbangan antara menangkap semua kasus positif (recall) dan menghindari prediksi positif yang salah (precision).

