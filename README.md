# Imam Agil Aiman - F55123066 - Teknik Informatika B
# Plat Nomor Classifier - Machine Learning UAS 2025

## Deskripsi Proyek
Aplikasi sederhana untuk mengenali asal daerah kendaraan berdasarkan awalan plat nomor. Model ini mengenali 4 kelas:
- **DN** → Sulawesi Tengah  
- **DD** → Sulawesi Selatan  
- **B**  → DKI Jakarta  
- **D**  → Bandung  

Data dummy digunakan dengan minimal **100 data per kelas**. Aplikasi ini **tidak menggunakan library full machine learning**, hanya menggunakan **NumPy dan modul Python standar** seperti `csv`, `re`, dan `random`.

---

## Metode
Model dibuat menggunakan algoritma **K-Nearest Neighbors (KNN)** dari nol, tanpa bantuan library eksternal.

Langkah-langkah:
1. Preprocessing plat nomor → konversi awalan jadi fitur numerik.
2. Training model dengan data dummy dari file `dataset_manual.csv`.
3. Testing akurasi model dengan data uji.
4. Input manual: pengguna bisa mengetik plat nomor dan melihat hasil prediksi asal daerahnya.

---

## Struktur File

├── dataset_manual.csv # Dataset dummy (400+ data, 4 kelas)

├── main.py # Kode utama aplikasi

├── README.md # Dokumentasi proyek


---

## Cara Menjalankan

1. Pastikan Python 3.x dan NumPy sudah terinstal.
2. Jalankan program dengan:
```bash
python main.py
