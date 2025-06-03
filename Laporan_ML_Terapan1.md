# Laporan Proyek Machine Learning Terapan | Diabetes Prediction

**Disusun oleh : Irfan Rizadi**
Ini adalah proyek pertama analisis prediktif untuk memenuhi submission Dicoding Kelas Machine Learning Terapan.
Proyek ini membangun model machine learning yang dapat memprediksi biaya pertanggungan medis tahunan.

## Domain Proyek

Diabetes merupakan salah satu penyakit kronis yang paling signifikan secara global, dengan prevalensi yang terus meningkat dari tahun ke tahun. Berdasarkan laporan dari Centers for Disease Control and Prevention (CDC), diabetes telah menjadi penyebab utama berbagai komplikasi kesehatan, termasuk penyakit jantung, stroke, gagal ginjal, dan amputasi anggota tubuh bagian bawah. Deteksi dini terhadap risiko diabetes menjadi sangat penting agar langkah-langkah pencegahan atau intervensi dapat dilakukan sejak awal.

Salah satu pendekatan untuk mendeteksi potensi risiko diabetes adalah dengan menganalisis indikator kesehatan (*health indicators*) yang dapat diperoleh melalui survei kesehatan masyarakat. Indikator ini meliputi faktor-faktor seperti indeks massa tubuh (BMI), tekanan darah tinggi, aktivitas fisik, konsumsi alkohol, status merokok, dan riwayat kesehatan keluarga. Pendekatan berbasis data memungkinkan pembangunan model prediksi yang dapat membantu tenaga medis atau individu untuk mengidentifikasi potensi risiko diabetes secara lebih akurat dan efisien.

Dalam studi CDC [1], digunakan data dari *Behavioral Risk Factor Surveillance System* (BRFSS) untuk menganalisis keterkaitan antara berbagai indikator kesehatan dengan kejadian diabetes pada populasi dewasa di Amerika Serikat. Studi ini menekankan pentingnya strategi pencegahan berbasis komunitas dan pendekatan individual dengan memanfaatkan data epidemiologis. Berdasarkan temuan ini, proyek ini bertujuan untuk membangun model prediksi diabetes menggunakan algoritma pembelajaran mesin (*machine learning*) yang dilatih pada data indikator kesehatan. Model ini diharapkan dapat menjadi alat bantu prediktif dalam upaya preventif, baik di ranah pelayanan kesehatan maupun sebagai sistem pendukung keputusan.

Dengan adanya sistem prediksi berbasis indikator kesehatan, upaya deteksi dini dapat diperluas dan lebih mudah diakses oleh masyarakat luas. Hal ini juga sejalan dengan arah transformasi digital di bidang kesehatan, yang menekankan pentingnya *data-driven decision making*.

**Referensi:**

[1] Centers for Disease Control and Prevention. (2017). *Prevalence and Incidence of Type 2 Diabetes and Prediabetes*. Morbidity and Mortality Weekly Report (MMWR), 66(43), 1201–1207. Tersedia di: [https://www.cdc.gov/mmwr/volumes/66/wr/mm6643a2.htm](https://www.cdc.gov/mmwr/volumes/66/wr/mm6643a2.htm)


## Business Understanding

Peningkatan jumlah penderita diabetes dari tahun ke tahun telah menjadi perhatian utama dalam dunia kesehatan. Banyak kasus diabetes yang terdiagnosis terlambat, sehingga menyebabkan komplikasi serius yang dapat dicegah apabila deteksi dilakukan lebih awal. Oleh karena itu, penting bagi institusi kesehatan, pembuat kebijakan, maupun masyarakat umum untuk memiliki alat bantu prediksi risiko diabetes yang akurat dan efisien berbasis data indikator kesehatan.

Proyek ini bertujuan untuk membangun sistem prediksi diabetes dengan memanfaatkan indikator kesehatan (health indicators) yang dapat dikumpulkan dari survei rutin atau data rekam medis sederhana. Model prediksi ini dapat digunakan dalam praktik preventif dan skrining awal untuk mendukung keputusan medis atau perencanaan kebijakan kesehatan.

### Problem Statements

- **Pernyataan Masalah 1:** Bagaimana mengidentifikasi individu dengan risiko tinggi terkena diabetes berdasarkan indikator kesehatan seperti BMI, aktivitas fisik, status merokok, dan riwayat tekanan darah tinggi?
- **Pernyataan Masalah 2:** Model prediksi seperti apa yang mampu memberikan hasil akurasi dan interpretabilitas terbaik dalam mengklasifikasikan risiko diabetes?
- **Pernyataan Masalah 3:** Bagaimana membandingkan efektivitas beberapa model machine learning dalam memprediksi risiko diabetes menggunakan data indikator kesehatan?

### Goals

- **Jawaban Masalah 1:** Mengembangkan sistem prediksi yang dapat mengklasifikasikan individu ke dalam kategori berisiko atau tidak berisiko diabetes berdasarkan data indikator kesehatan.
- **Jawaban Masalah 2:** Menentukan model terbaik dari beberapa algoritma klasifikasi berdasarkan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.
- **Jawaban Masalah 3:** Melakukan analisis perbandingan performa antara model Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, dan CatBoost dalam memprediksi diabetes.

### Solution Statements

- **Solusi 1:** Membangun dan melatih beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, Decision Tree dan XGBoostb untuk memprediksi risiko diabetes.
- **Solusi 2:** Melakukan hyperparameter tuning pada setiap model untuk meningkatkan akurasi dan performa klasifikasi.
- **Solusi 3:** Menggunakan metrik evaluasi seperti accuracy, precision, recall, dan F1-score untuk membandingkan performa antar model secara objektif.


# Data Understanding

Dataset yang digunakan dalam proyek ini adalah CDC Diabetes Health Indicators yang diperoleh dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). Dataset ini berisi indikator kesehatan yang terkait dengan diabetes, prediabetes, dan faktor-faktor risiko lainnya yang dikumpulkan oleh Centers for Disease Control and Prevention (CDC). Data ini sangat berguna untuk memahami faktor-faktor yang berkontribusi terhadap risiko diabetes dalam populasi.

## Variabel-variabel pada CDC Diabetes Health Indicators dataset adalah sebagai berikut:

**Target Variable:**
- **Diabetes_binary**: Variabel target biner yang menunjukkan status diabetes (0 = tidak diabetes, 1 = prediabetes atau diabetes)

**Health Condition Features:**
- **HighBP**: Tekanan darah tinggi (0 = tidak memiliki hipertensi, 1 = memiliki hipertensi)
- **HighChol**: Kolesterol tinggi (0 = tidak memiliki kolesterol tinggi, 1 = memiliki kolesterol tinggi)
- **CholCheck**: Pemeriksaan kolesterol dalam 5 tahun terakhir (0 = tidak melakukan pemeriksaan, 1 = melakukan pemeriksaan)
- **Stroke**: Riwayat stroke (0 = tidak pernah stroke, 1 = pernah stroke)
- **HeartDiseaseorAttack**: Riwayat penyakit jantung koroner atau serangan jantung (0 = tidak ada riwayat, 1 = ada riwayat)

**Physical Health Features:**
- **BMI**: Indeks Massa Tubuh (Body Mass Index) dalam bentuk nilai integer
- **PhysHlth**: Jumlah hari kesehatan fisik buruk dalam 30 hari terakhir (skala 1-30 hari)
- **DiffWalk**: Kesulitan berjalan atau menaiki tangga (0 = tidak ada kesulitan, 1 = ada kesulitan serius)

**Lifestyle Features:**
- **Smoker**: Status merokok - pernah merokok minimal 100 batang sepanjang hidup (0 = tidak, 1 = ya)
- **PhysActivity**: Aktivitas fisik dalam 30 hari terakhir diluar pekerjaan (0 = tidak ada, 1 = ada)
- **Fruits**: Konsumsi buah 1 kali atau lebih per hari (0 = tidak, 1 = ya)
- **Veggies**: Konsumsi sayuran 1 kali atau lebih per hari (0 = tidak, 1 = ya)
- **HvyAlcoholConsump**: Konsumsi alkohol berat (pria >14 gelas/minggu, wanita >7 gelas/minggu) (0 = tidak, 1 = ya)

**Mental Health Features:**
- **MentHlth**: Jumlah hari kesehatan mental buruk dalam 30 hari terakhir (skala 1-30 hari)
- **GenHlth**: Penilaian kesehatan umum (skala 1-5: 1 = sangat baik, 2 = baik, 3 = cukup, 4 = buruk, 5 = sangat buruk)

**Healthcare Access Features:**
- **AnyHealthcare**: Memiliki akses layanan kesehatan termasuk asuransi (0 = tidak ada, 1 = ada)
- **NoDocbcCost**: Pernah tidak bisa ke dokter karena masalah biaya dalam 12 bulan terakhir (0 = tidak pernah, 1 = pernah)

**Demographic Features:**
- **Sex**: Jenis kelamin (0 = perempuan, 1 = laki-laki)
- **Age**: Kategori usia 13 tingkat (1 = 18-24 tahun, 9 = 60-64 tahun, 13 = 80 tahun atau lebih)
- **Education**: Tingkat pendidikan (skala 1-6: 1 = tidak pernah sekolah, 6 = sarjana atau lebih)
- **Income**: Tingkat pendapatan (skala 1-8: 1 = kurang dari $10,000, 8 = $75,000 atau lebih)

## Analisis Univariate - Sebaran Data Kolom

### Health Condition Features

**HighBP (Tekanan Darah Tinggi)**
Distribusi biner dari data HighBP menunjukkan bahwa hanya terdapat dua nilai (0 = tidak hipertensi, 1 = hipertensi). Berdasarkan perbandingan frekuensi, lebih banyak individu yang tidak mengalami tekanan darah tinggi. Sebaran data tidak menunjukkan outlier yang signifikan dan berbentuk bimodal. Kesimpulan dari analisis ini adalah mayoritas individu dalam dataset tidak memiliki tekanan darah tinggi.

**HighChol (Kolesterol Tinggi)**
Data HighChol memiliki distribusi biner dengan dua kategori utama (0 = tidak memiliki kolesterol tinggi, 1 = memiliki kolesterol tinggi). Perbandingan frekuensi menunjukkan bahwa lebih banyak individu memiliki kadar kolesterol normal dibandingkan tinggi. Boxplot menunjukkan persebaran data tanpa outlier signifikan, sementara violin plot mengonfirmasi pola distribusi yang simetris. Mayoritas individu dalam dataset tidak memiliki kadar kolesterol tinggi.

**CholCheck (Pemeriksaan Kolesterol)**
Distribusi biner dari data CholCheck terdiri dari dua kategori utama (0 = tidak melakukan pemeriksaan kolesterol, 1 = melakukan pemeriksaan kolesterol). Histogram menunjukkan bahwa lebih banyak individu telah melakukan pemeriksaan kolesterol dibandingkan yang belum. Data tersebar dalam dua kelompok utama tanpa outlier signifikan, dengan distribusi berbentuk biner. Mayoritas individu dalam dataset telah melakukan pemeriksaan kadar kolesterol mereka.

**Stroke (Riwayat Stroke)**
Data Stroke memiliki distribusi biner dengan dua kategori utama (0 = tidak mengalami stroke, 1 = mengalami stroke). Histogram menunjukkan bahwa jumlah individu yang tidak mengalami stroke jauh lebih banyak dibandingkan mereka yang mengalami stroke. Persebaran data tidak menunjukkan outlier yang signifikan, dengan distribusi yang memiliki puncak jelas di dua kategori utama. Mayoritas individu dalam dataset tidak mengalami stroke, namun terdapat proporsi kecil yang terdampak.

**HeartDiseaseorAttack (Penyakit Jantung atau Serangan Jantung)**
Distribusi biner dari data HeartDiseaseorAttack terdiri dari dua kategori utama (0 = tidak mengalami penyakit jantung atau serangan jantung, 1 = mengalami penyakit jantung atau serangan jantung). Jumlah individu yang tidak mengalami penyakit jantung jauh lebih banyak dibandingkan mereka yang terdampak. Data tersebar tanpa outlier yang signifikan, dengan distribusi yang memiliki puncak jelas di dua kategori utama. Mayoritas individu dalam dataset tidak mengalami penyakit jantung atau serangan jantung.

### Physical Health Features

**BMI (Indeks Massa Tubuh)**
Data BMI memiliki distribusi kontinu dengan nilai yang beragam, bukan kategori biner. Histogram menunjukkan bahwa mayoritas individu memiliki BMI dalam rentang normal hingga obesitas ringan. Boxplot mengindikasikan keberadaan beberapa outlier, terutama di nilai BMI yang sangat tinggi, sementara violin plot menunjukkan distribusi dengan puncak pada kategori overweight hingga obesitas. Sebagian besar individu dalam dataset memiliki BMI di kategori overweight atau obesitas, dengan beberapa outlier yang menunjukkan kasus ekstrem.

**PhysHlth (Kesehatan Fisik)**
Data PhysHlth memiliki distribusi kontinu dengan rentang nilai yang lebih luas, menunjukkan jumlah hari kesehatan fisik buruk selama sebulan terakhir. Histogram menunjukkan sebagian besar individu mengalami sedikit atau tidak ada hari kesehatan fisik buruk dalam sebulan. Boxplot menunjukkan keberadaan beberapa outlier dengan nilai yang lebih tinggi, sementara violin plot mengonfirmasi distribusi yang condong ke nilai rendah. Mayoritas individu mengalami sedikit atau tidak ada masalah kesehatan fisik selama sebulan terakhir.

**DiffWalk (Kesulitan Berjalan)**
Distribusi biner dari data DiffWalk terdiri dari dua kategori utama (0 = tidak mengalami kesulitan berjalan, 1 = mengalami kesulitan berjalan). Jumlah individu yang tidak mengalami kesulitan berjalan jauh lebih banyak dibandingkan mereka yang mengalami kesulitan. Data tersebar tanpa outlier yang signifikan, dengan pola distribusi yang memiliki dua puncak jelas. Mayoritas individu dalam dataset tidak mengalami kesulitan berjalan.

### Lifestyle Features

**Smoker (Status Merokok)**
Data Smoker memiliki distribusi biner dengan dua kategori utama (0 = tidak merokok, 1 = merokok). Histogram menunjukkan bahwa jumlah individu yang tidak merokok lebih banyak dibandingkan perokok. Data tersebar tanpa outlier signifikan, dengan pola distribusi yang jelas dengan dua puncak. Mayoritas individu dalam dataset bukan perokok, namun terdapat proporsi signifikan yang merupakan perokok aktif.

**PhysActivity (Aktivitas Fisik)**
Distribusi biner dari data PhysActivity terdiri dari dua kategori utama (0 = tidak melakukan aktivitas fisik, 1 = melakukan aktivitas fisik). Lebih banyak individu yang melakukan aktivitas fisik dibandingkan yang tidak. Data tersebar tanpa outlier signifikan, dengan pola distribusi yang jelas dengan dua puncak. Mayoritas individu dalam dataset melakukan aktivitas fisik secara rutin.

**Fruits (Konsumsi Buah)**
Data Fruits memiliki distribusi biner dengan dua kategori utama (0 = tidak mengonsumsi buah secara rutin, 1 = mengonsumsi buah secara rutin). Lebih banyak individu yang rutin mengonsumsi buah dibandingkan yang tidak. Data tersebar tanpa outlier signifikan, dengan distribusi berbentuk biner. Mayoritas individu dalam dataset mengonsumsi buah secara rutin sebagai bagian dari pola makan mereka.

**Veggies (Konsumsi Sayuran)**
Distribusi biner dari data Veggies terdiri dari dua kategori utama (0 = tidak mengonsumsi sayuran secara rutin, 1 = mengonsumsi sayuran secara rutin). Lebih banyak individu yang rutin mengonsumsi sayuran dibandingkan yang tidak. Data tersebar tanpa outlier signifikan, dengan distribusi berbentuk biner dengan dua puncak utama. Mayoritas individu memiliki kebiasaan mengonsumsi sayuran secara rutin sebagai bagian dari pola makan mereka.

**HvyAlcoholConsump (Konsumsi Alkohol Berat)**
Data HvyAlcoholConsump memiliki distribusi biner dengan dua kategori utama (0 = tidak mengonsumsi alkohol dalam jumlah besar, 1 = mengonsumsi alkohol dalam jumlah besar). Jumlah individu yang tidak mengonsumsi alkohol dalam jumlah besar lebih banyak dibandingkan mereka yang melakukannya. Mayoritas individu dalam dataset tidak mengonsumsi alkohol secara berlebihan.

### Healthcare Access Features

**AnyHealthcare (Akses Layanan Kesehatan)**
Distribusi biner dari data AnyHealthcare terdiri dari dua kategori utama (0 = tidak memiliki akses layanan kesehatan, 1 = memiliki akses layanan kesehatan). Histogram menunjukkan bahwa mayoritas individu memiliki akses terhadap layanan kesehatan. Data tersebar tanpa outlier signifikan, dengan pola distribusi yang jelas dengan dua puncak. Mayoritas individu dalam dataset memiliki akses terhadap layanan kesehatan, meskipun ada sebagian kecil yang tidak.

**NoDocbcCost (Kendala Biaya Layanan Kesehatan)**
Data NoDocbcCost memiliki distribusi biner dengan dua kategori utama (0 = tidak mengalami kendala biaya untuk layanan kesehatan, 1 = mengalami kendala biaya untuk layanan kesehatan). Jumlah individu yang tidak mengalami kendala biaya lebih banyak dibandingkan yang mengalami kesulitan. Data tersebar tanpa outlier signifikan, dengan pola distribusi yang jelas dengan dua puncak. Mayoritas individu tidak mengalami kendala biaya dalam mengakses layanan kesehatan, namun terdapat proporsi signifikan yang menghadapi hambatan finansial.

### Demographic Features

**Sex (Jenis Kelamin)**
Data Sex memiliki distribusi biner dengan dua kategori utama (0 = perempuan, 1 = laki-laki). Histogram menunjukkan bahwa jumlah perempuan lebih banyak dibandingkan laki-laki dalam dataset. Distribusi seimbang tanpa outlier signifikan, dengan pola distribusi yang jelas dengan dua puncak utama. Mayoritas individu dalam dataset adalah perempuan, namun distribusinya tetap cukup seimbang antara kedua kategori.

**Age (Usia)**
Data Age memiliki distribusi kategorikal yang dikategorikan dalam beberapa kelompok usia dengan nilai numerik. Histogram menunjukkan distribusi yang cenderung lebih banyak pada kelompok usia menengah hingga lanjut. Data tersebar dengan rentang yang cukup luas tanpa outlier signifikan, dengan distribusi yang berpola. Mayoritas individu berasal dari kelompok usia menengah hingga lanjut, dengan distribusi yang merata di berbagai kategori usia.

**Education (Tingkat Pendidikan)**
Distribusi kategorikal dari data Education terdiri dari beberapa kategori yang mencerminkan tingkat pendidikan individu. Histogram menunjukkan distribusi tingkat pendidikan, dengan beberapa kategori lebih dominan dibandingkan lainnya. Data tersebar tanpa outlier signifikan, dengan distribusi yang memiliki pola tertentu. Mayoritas individu memiliki tingkat pendidikan yang bervariasi, dengan distribusi yang mencerminkan karakteristik populasi.

**Income (Tingkat Pendapatan)**
Data Income memiliki distribusi kategorikal yang dikategorikan dalam beberapa kelompok yang mencerminkan tingkat pendapatan individu. Histogram menunjukkan distribusi tingkat pendapatan, dengan beberapa kategori lebih dominan dibandingkan lainnya. Data tersebar tanpa outlier signifikan, dengan pola distribusi yang memiliki variasi jelas. Mayoritas individu berasal dari kelompok pendapatan tertentu, dengan distribusi yang mencerminkan karakteristik ekonomi populasi.

## Exploratory Data Analysis (Analisis Multivariate)

Untuk memahami karakteristik data lebih mendalam, dilakukan beberapa tahapan analisis eksplorasi data:

**1. Analisis Distribusi Variabel**

![Scatter Plot Matrix Variabel Kesehatan](https://github.com/irfnriza/Diabetes-Predictions/blob/main/assets/plot0.png)

Visualisasi scatter plot matrix menunjukkan distribusi dan hubungan antar variabel kesehatan utama. Dapat diamati bahwa sebagian besar variabel memiliki distribusi biner yang jelas, dengan BMI sebagai satu-satunya variabel kontinu yang menunjukkan distribusi normal dengan beberapa outlier.

**2. Analisis Distribusi Variabel Gaya Hidup**

![Scatter Plot Matrix Kebiasaan Hidup](https://github.com/irfnriza/Diabetes-Predictions/blob/main/assets/plot1.png?raw=true)

Matrix ini memperlihatkan distribusi variabel-variabel terkait gaya hidup seperti merokok, aktivitas fisik, konsumsi buah dan sayuran, serta konsumsi alkohol. Mayoritas individu menunjukkan gaya hidup yang relatif sehat dengan aktivitas fisik yang cukup dan konsumsi buah-sayuran yang baik.

**3. Analisis Korelasi - Variabel Kesehatan**

![Correlation Heatmap Diabetes dengan Variabel Kesehatan](https://github.com/irfnriza/Diabetes-Predictions/blob/main/assets/heatmaps0.png?raw=true)

Heatmap korelasi menunjukkan hubungan diabetes dengan berbagai kondisi kesehatan. Terlihat bahwa diabetes memiliki korelasi positif yang cukup kuat dengan tekanan darah tinggi (0.26), kesehatan umum yang buruk (0.29), dan usia (0.18). BMI juga menunjukkan korelasi positif (0.22) dengan diabetes.

**4. Analisis Korelasi - Variabel Gaya Hidup**

![Correlation Heatmap Diabetes dengan Variabel Kebiasaan Hidup](https://github.com/irfnriza/Diabetes-Predictions/blob/main/assets/heatmaps1.png?raw=true)

Korelasi diabetes dengan variabel gaya hidup menunjukkan bahwa kesulitan berjalan memiliki korelasi positif tertinggi (0.22) dengan diabetes. Menariknya, aktivitas fisik menunjukkan korelasi negatif (-0.12), mengindikasikan bahwa aktivitas fisik dapat bersifat protektif terhadap diabetes.

**5. Analisis Komponen Utama (PCA)**

![PCA Projection 2D](https://github.com/irfnriza/Diabetes-Predictions/blob/main/assets/pca.png?raw=true)

Visualisasi PCA 2D menunjukkan pemisahan yang cukup jelas antara individu dengan diabetes (biru) dan tanpa diabetes (merah) dalam ruang komponen utama. Hal ini mengindikasikan bahwa kombinasi variabel-variabel dalam dataset dapat membedakan kedua kelompok dengan cukup baik, yang menjadi dasar untuk pengembangan model prediktif.

Berdasarkan analisis eksplorasi data ini, dapat disimpulkan bahwa dataset memiliki kualitas yang baik untuk pengembangan model prediksi diabetes, dengan variabel-variabel yang menunjukkan pola dan korelasi yang bermakna secara medis.

## Data Preparation

Pada bagian ini dilakukan beberapa teknik data preparation yang penting untuk mempersiapkan dataset sebelum tahap pemodelan. Teknik-teknik yang diterapkan secara berurutan adalah sebagai berikut:

### 1. Menghapus Data Duplikat

**Proses yang dilakukan:**
```python
df.drop_duplicates(inplace=True)
```

**Penjelasan:**
Tahap ini bertujuan untuk menghapus baris yang memiliki data yang sama secara keseluruhan dalam dataset. Proses ini dilakukan untuk memperbaiki kualitas data dengan menghilangkan redundansi yang tidak diperlukan.

**Alasan diperlukan:**
- Menghemat ruang penyimpanan dan meningkatkan efisiensi komputasi
- Mencegah bias dalam model yang dapat terjadi akibat data yang berulang
- Memastikan setiap sampel dalam dataset adalah unik dan representatif


### 2. Menghapus Nilai Outlier

**Proses yang dilakukan:**
```python
df = df[(df['BMI'] >= 13.5) & (df['BMI'] <= 41.5)]
```

**Penjelasan:**
Tahap ini menghapus nilai ekstrem dalam kolom BMI yang berada di luar batas wajar. Batas bawah (13.5) dan batas atas (41.5) ditentukan menggunakan metode **Interquartile Range (IQR)**.

**Alasan diperlukan:**
- Meningkatkan akurasi model dengan menghilangkan nilai yang dapat mengganggu analisis data
- Memastikan data BMI berada dalam rentang yang secara medis masuk akal
- Mencegah model overfitting pada nilai-nilai ekstrem yang tidak representatif


### 3. Seleksi Fitur dengan SelectKBest

**Proses yang dilakukan:**
Pemilihan fitur terbaik berdasarkan F-score untuk meningkatkan performa model prediksi diabetes.

**Penjelasan:**
Berdasarkan hasil analisis SelectKBest, fitur-fitur diprioritaskan berdasarkan tingkat signifikansinya:

**Fitur paling signifikan:**
- GenHlth, HighBP, DiffWalk, BMI, dan HighChol

**Fitur dengan pengaruh sedang:**
- Age, HeartDiseaseorAttack, PhysHlth, Income, dan Education

**Fitur dengan pengaruh rendah:**
- Smoker, Veggies, Sex, AnyHealthcare, Fruits, dan NoDocbcCost

**Alasan diperlukan:**
- Mengurangi dimensionalitas data dan kompleksitas model
- Meningkatkan performa model dengan fokus pada fitur yang paling berpengaruh
- Menghindari curse of dimensionality dan overfitting
- Semua fitur memiliki p-value ≈ 0, menunjukkan kontribusi signifikan dalam membedakan individu dengan dan tanpa diabetes


### 4. Pembagian Data Training dan Testing

**Proses yang dilakukan:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

**Penjelasan:**
Dataset dibagi menjadi data training (80%) dan testing (20%) dengan stratifikasi untuk mempertahankan proporsi kelas.

Alasan diperlukan:
- Memisahkan data untuk evaluasi yang objektif
- Parameter `stratify=y` memastikan proporsi kelas seimbang antara training dan testing
- `random_state=42` digunakan untuk reproduksibilitas hasil
- **Hasil:** Training set berisi 300,640 sampel, Testing set berisi 43,926 sampel


### 5. Penerapan SMOTE untuk Mengatasi Ketidakseimbangan Kelas

**Proses yang dilakukan:**
```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

Penjelasan:
**SMOTE (Synthetic Minority Oversampling Technique)** diterapkan untuk mengatasi ketidakseimbangan kelas dengan menghasilkan sampel sintetis dari kelas minoritas menggunakan pendekatan nearest neighbors.

**Alasan diperlukan:**
- Dataset awal mengalami ketidakseimbangan kelas yang signifikan
- **Setelah SMOTE:** kelas positif dan negatif menjadi seimbang (masing-masing 150,320 sampel)
- Meningkatkan kemampuan model untuk memprediksi kelas minoritas (diabetes)
- Data testing tetap mempertahankan distribusi alami (86% non-diabetes, 14% diabetes) untuk evaluasi realistis


### 6. Scaling Fitur Numerik dengan RobustScaler

**Proses yang dilakukan:**
```python
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
```

### Penjelasan:
Fitur numerik (`GenHlth`, `BMI`, `Age`, `PhysHlth`, `Income`, dan `Education`) dinormalisasi menggunakan **RobustScaler** yang berbasis median dan Interquartile Range (IQR).

### Alasan diperlukan:
- RobustScaler lebih tahan terhadap outlier dibandingkan StandardScaler atau MinMaxScaler
- Memastikan semua fitur memiliki skala yang sebanding untuk algoritma machine learning
- Scaling hanya dilakukan pada data training yang telah melalui SMOTE, kemudian diterapkan pada data testing
- Mencegah data leakage dengan melakukan fit hanya pada data training


### Hasil Akhir Data Preparation

Setelah seluruh tahapan data preparation, dataset siap untuk tahap pemodelan dengan karakteristik:

**Data bersih** tanpa duplikat dan outlier  
**Fitur terpilih** berdasarkan signifikansi statistik  
**Kelas seimbang** pada data training  
**Skala fitur** yang konsisten dan robust terhadap outlier


## Modeling
Tahapan modeling dalam proyek ini mengimplementasikan empat algoritma machine learning yang berbeda untuk memprediksi risiko diabetes berdasarkan data kesehatan yang telah dipreprocessing. Setiap algoritma dipilih berdasarkan karakteristik uniknya dalam menangani dataset tabular dan kemampuan interpretasinya untuk aplikasi medis.

### Algoritma yang Digunakan

#### 1. Logistic Regression
**Kelebihan:**
- Model linear yang sederhana dan mudah diinterpretasi
- Cepat dalam training dan prediksi
- Memberikan probabilitas output yang berguna untuk decision making medis
- Tidak memerlukan feature scaling yang rumit
- Robust terhadap outlier

**Kekurangan:**
- Terbatas pada hubungan linear antar fitur
- Performa menurun pada data dengan kompleksitas tinggi
- Sensitif terhadap multicollinearity
- Memerlukan dataset yang cukup besar untuk stabilitas

#### 2. Random Forest
**Kelebihan:**
- Menangani hubungan non-linear dan interaksi kompleks antar fitur
- Robust terhadap overfitting melalui ensemble method
- Menyediakan feature importance yang berguna untuk interpretasi medis
- Dapat menangani missing values dan outlier dengan baik
- Tidak memerlukan feature scaling

**Kekurangan:**
- Model ensemble yang relatif kompleks dan sulit diinterpretasi secara detail
- Memerlukan memory yang lebih besar
- Dapat bias terhadap fitur kategorikal dengan banyak level
- Training time lebih lama dibanding model sederhana

#### 3. Decision Tree
**Kelebihan:**
- Sangat mudah diinterpretasi dan divisualisasikan
- Dapat menangani fitur kategorikal dan numerik tanpa preprocessing
- Menghasilkan rule-based decision yang mudah dipahami tenaga medis
- Cepat dalam prediksi

**Kekurangan:**
- Sangat rentan terhadap overfitting
- Tidak stabil (small changes in data dapat mengubah struktur tree)
- Bias terhadap fitur dengan range nilai yang besar
- Performa terbatas pada dataset kompleks

#### 4. XGBoost
**Kelebihan:**
- Algoritma boosting yang powerful dengan performa tinggi
- Mengoptimalkan bias-variance tradeoff melalui gradient boosting
- Built-in regularization untuk mencegah overfitting
- Menangani missing values secara otomatis
- Feature importance yang akurat

**Kekurangan:**
- Memerlukan hyperparameter tuning yang ekstensif
- Rentan terhadap overfitting jika tidak di-tune dengan baik
- Kompleks dan sulit diinterpretasi
- Sensitif terhadap outlier dan noise

### Proses Hyperparameter Tuning

Untuk meningkatkan performa setiap model, dilakukan hyperparameter tuning menggunakan GridSearchCV dengan 5-fold cross validation. Parameter yang di-tune untuk setiap algoritma:

**Logistic Regression:**
- `C`: [0.01, 0.1, 1, 10] - Regularization strength
- `penalty`: ['l2'] - Regularization type
- `solver`: ['lbfgs'] - Optimization algorithm

**Random Forest:**
- `n_estimators`: [100, 200] - Jumlah trees
- `max_depth`: [None, 10, 20] - Kedalaman maksimum tree
- `min_samples_split`: [2, 5] - Minimum samples untuk split

**Decision Tree:**
- `max_depth`: [None, 5, 10, 20] - Kedalaman maksimum
- `criterion`: ['gini', 'entropy'] - Splitting criterion

**XGBoost:**
- `n_estimators`: [100, 200] - Jumlah boosting rounds
- `max_depth`: [3, 6, 9] - Kedalaman tree
- `learning_rate`: [0.01, 0.1, 0.2] - Step size shrinkage
- `subsample`: [0.8, 1.0] - Sampling ratio
- `colsample_bytree`: [0.8, 1.0] - Feature sampling ratio

### Hasil Evaluasi Model

| Model | Akurasi Baseline | Akurasi Setelah Tuning | Peningkatan | Parameter Terbaik |
|-------|------------------|------------------------|-------------|-------------------|
| **Random Forest** | 0.7455 | **0.7463** | +0.0008 | `n_estimators=200, max_depth=None, min_samples_split=2` |
| **XGBoost** | 0.6928 | **0.7217** | +0.0289 | `n_estimators=200, max_depth=9, learning_rate=0.2, subsample=0.8, colsample_bytree=1.0` |
| **Decision Tree** | 0.7207 | **0.7207** | 0.0000 | `criterion=gini, max_depth=None` |
| **Logistic Regression** | 0.6958 | **0.6957** | -0.0001 | `C=0.01, penalty=l2, solver=lbfgs` |

### Pemilihan Model Terbaik

**Random Forest dipilih sebagai model terbaik** berdasarkan pertimbangan berikut:

1. **Akurasi Tertinggi**: Mencapai 74.63% setelah hyperparameter tuning, mengungguli model lainnya
2. **Konsistensi Performa**: Peningkatan akurasi yang stabil meski kecil, menunjukkan robustness model
3. **Interpretabilitas**: Menyediakan feature importance yang dapat membantu tenaga medis memahami faktor-faktor penting dalam prediksi diabetes
4. **Robustness**: Ensemble method yang mengurangi risiko overfitting dibanding Decision Tree tunggal
5. **Aplikabilitas Medis**: Balance yang baik antara akurasi dan interpretabilitas, cocok untuk decision support system di bidang kesehatan

**Alasan Penolakan Model Lain:**
- **XGBoost**: Meski mengalami peningkatan signifikan (69.28% → 72.17%), masih kalah akurasi dari Random Forest dan lebih sulit diinterpretasi
- **Decision Tree**: Tidak menunjukkan improvement dan rentan overfitting
- **Logistic Regression**: Akurasi terendah dan terbatas pada hubungan linear

### Proses Improvement yang Dilakukan

1. **Systematic Hyperparameter Search**: Menggunakan GridSearchCV untuk mencari kombinasi parameter optimal
2. **Cross-Validation**: Implementasi 5-fold CV untuk evaluasi yang robust
3. **Parameter Range Selection**: Pemilihan range parameter berdasarkan best practices dan karakteristik dataset
4. **Performance Tracking**: Monitoring peningkatan akurasi untuk setiap model

**Hasil Improvement:**
- XGBoost menunjukkan improvement terbesar (+2.89%), mengkonfirmasi pentingnya tuning untuk algoritma boosting
- Random Forest tetap stabil dengan sedikit peningkatan, menunjukkan robustness hyperparameter default
- Model linear (Logistic Regression) dan Decision Tree tidak menunjukkan improvement signifikan, mengindikasikan keterbatasan algoritma untuk kompleksitas data ini

Model Random Forest yang telah di-tune akan digunakan untuk tahap evaluasi selanjutnya menggunakan metrik comprehensive seperti precision, recall, F1-score, dan analisis confusion matrix.

# Evaluation

Pada bagian ini akan dijelaskan metrik evaluasi yang digunakan untuk mengukur performa model prediksi diabetes, serta analisis hasil berdasarkan metrik tersebut.

## Metrik Evaluasi yang Digunakan

### 1. AUC-ROC (Area Under the Curve - Receiver Operating Characteristic)

**Formula dan Cara Kerja:**
AUC-ROC mengukur kemampuan model untuk membedakan antara kelas positif (diabetes) dan kelas negatif (non-diabetes). Kurva ROC dibuat dengan memplot True Positive Rate (TPR) terhadap False Positive Rate (FPR) pada berbagai threshold.

- **True Positive Rate (TPR) = Sensitivity/Recall = TP / (TP + FN)**
- **False Positive Rate (FPR) = 1 - Specificity = FP / (FP + TN)**
- **AUC Score berkisar dari 0.5 (random classifier) hingga 1.0 (perfect classifier)**

**Interpretasi AUC Score:**
- **0.9 - 1.0**: Excellent
- **0.8 - 0.9**: Good  
- **0.7 - 0.8**: Fair
- **0.6 - 0.7**: Poor
- **0.5 - 0.6**: Very Poor

### 2. Accuracy (Akurasi)

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Akurasi mengukur proporsi prediksi yang benar dari total prediksi. Namun, dalam kasus ketidakseimbangan kelas seperti prediksi diabetes, akurasi saja tidak cukup untuk mengevaluasi performa model.

### 3. Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

Precision mengukur dari semua prediksi positif (diabetes), berapa banyak yang benar-benar positif. Metrik ini penting untuk mengurangi false positive dalam diagnosis medis.

### 4. Recall (Sensitivity)

**Formula:**
```
Recall = TP / (TP + FN)
```

Recall mengukur dari semua kasus diabetes yang sebenarnya, berapa banyak yang berhasil dideteksi oleh model. Dalam konteks medis, recall tinggi sangat penting untuk tidak melewatkan pasien diabetes.

### 5. F1-Score

**Formula:**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

F1-Score adalah harmonic mean dari precision dan recall, memberikan keseimbangan antara kedua metrik tersebut.

---

## Hasil Evaluasi Model

### Performa Model Basic (Sebelum Tuning)

![Plot Evaluasi Model Basic](https://github.com/irfnriza/Diabetes-Predictions/blob/main/assets/evaluasi0.png?raw=true)

Berdasarkan kurva ROC untuk model basic, berikut adalah hasil AUC-ROC:

| Model | AUC-ROC Score | Kategori Performa |
|-------|---------------|-------------------|
| **Logistic Regression** | **0.78** | **Fair** |
| **Random Forest** | **0.71** | **Fair** |
| **Decision Tree** | **0.59** | **Very Poor** |
| **XGBoost** | **0.77** | **Fair** |

**Analisis Model Basic:**
- **Logistic Regression** menunjukkan performa terbaik dengan AUC-ROC 0.78, mengindikasikan kemampuan yang cukup baik dalam membedakan antara individu diabetes dan non-diabetes
- **XGBoost** menempati posisi kedua dengan AUC-ROC 0.77, menunjukkan performa yang hampir setara dengan Logistic Regression
- **Random Forest** memiliki AUC-ROC 0.71, masih dalam kategori fair namun lebih rendah dari model linear
- **Decision Tree** menunjukkan performa terburuk dengan AUC-ROC 0.59, hampir mendekati random classifier

### Performa Model Setelah Tuning

![Plot Evaluasi Model Hyperparameter Tuning](https://github.com/irfnriza/Diabetes-Predictions/blob/main/assets/evaluasi1.png?raw=true)

Setelah dilakukan hyperparameter tuning, berikut adalah perubahan performa:

| Model | AUC-ROC Basic | AUC-ROC Tuned | Peningkatan |
|-------|---------------|---------------|-------------|
| **Logistic Regression** | **0.78** | **0.78** | **Konsisten** |
| **Random Forest** | **0.71** | **0.72** | **+0.01** |
| **Decision Tree** | **0.59** | **0.59** | **Konsisten** |
| **XGBoost** | **0.77** | **0.74** | **-0.03** |

**Analisis Setelah Tuning:**
- **Logistic Regression** tetap mempertahankan performa terbaiknya dengan AUC-ROC 0.78, menunjukkan stabilitas model
- **Random Forest** mengalami sedikit peningkatan dari 0.71 menjadi 0.72 setelah tuning
- **Decision Tree** tidak mengalami perubahan signifikan, tetap pada AUC-ROC 0.59
- **XGBoost** justru mengalami sedikit penurunan dari 0.77 menjadi 0.74, kemungkinan karena overfitting setelah tuning

---

## Interpretasi Hasil Berdasarkan Problem Statement

### Jawaban terhadap Problem Statement 1
*"Bagaimana mengidentifikasi individu dengan risiko tinggi terkena diabetes berdasarkan indikator kesehatan?"*

**Hasil:** Model **Logistic Regression** dengan AUC-ROC 0.78 menunjukkan kemampuan yang cukup baik dalam mengidentifikasi individu berisiko diabetes. Kurva ROC menunjukkan bahwa model dapat membedakan antara individu diabetes dan non-diabetes dengan tingkat akurasi yang reasonable.

### Jawaban terhadap Problem Statement 2
*"Model prediksi seperti apa yang mampu memberikan hasil akurasi dan interpretabilitas terbaik?"*

**Hasil:** **Logistic Regression** menjadi model terbaik karena:
- Memiliki AUC-ROC tertinggi (0.78) baik sebelum maupun setelah tuning
- Memberikan interpretabilitas yang tinggi melalui koefisien linear
- Stabil dalam performa tanpa overfitting
- Cocok untuk aplikasi medis yang membutuhkan transparansi dalam pengambilan keputusan

### Jawaban terhadap Problem Statement 3
*"Bagaimana membandingkan efektivitas beberapa model machine learning?"*

**Hasil Perbandingan:**
1. **Logistic Regression**: Model terbaik dengan konsistensi tinggi
2. **XGBoost**: Performa baik namun mengalami penurunan setelah tuning
3. **Random Forest**: Performa sedang dengan sedikit peningkatan setelah tuning
4. **Decision Tree**: Performa terburuk, tidak cocok untuk dataset ini

---

## Kesimpulan Evaluasi

### Model Terpilih
**Logistic Regression** dipilih sebagai model terbaik untuk prediksi diabetes berdasarkan:
- **AUC-ROC Score**: 0.78 (kategori Fair)
- **Stabilitas**: Performa konsisten sebelum dan setelah tuning
- **Interpretabilitas**: Mudah dipahami oleh praktisi medis
- **Efisiensi**: Computational cost rendah

### Rekomendasi Implementasi
1. **Model Logistic Regression** dapat diimplementasikan sebagai tool skrining awal diabetes
2. Dengan AUC-ROC 0.78, model ini memberikan akurasi yang cukup untuk identifikasi risiko tinggi
3. Perlu dilakukan validasi lebih lanjut dengan data real-world sebelum implementasi klinis
4. Model dapat diintegrasikan dalam sistem informasi kesehatan untuk mendukung keputusan preventif

### Limitasi
- AUC-ROC 0.78 masih dalam kategori "Fair", sehingga masih ada ruang untuk perbaikan
- Model perlu dikombinasikan dengan penilaian klinis profesional
- Diperlukan monitoring berkelanjutan terhadap performa model di lapangan
