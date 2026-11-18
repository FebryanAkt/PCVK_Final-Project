# VisionLab Studio - Klasifikasi Bentuk Wajah Berbasis Jurnal Ilmiah
Aplikasi Pengolahan Citra Digital dan Visi Komputer yang disesuaikan secara langsung dengan metodologi penelitian pada jurnal:
**"Human Face Shape Classification with Machine Learning"** (Ashwinee Mehta & Taha Mohamed Mahmoud).

---

## Ringkasan Metodologi Jurnal

Penelitian ini mengklasifikasikan bentuk wajah manusia ke dalam 5 kategori geometris:
1. **Heart** (Bentuk Hati / Segitiga Terbalik)
2. **Oblong** (Bentuk Persegi Panjang / Memanjang)
3. **Oval** (Bentuk Lonjong Seimbang)
4. **Round** (Bentuk Bulat)
5. **Square** (Bentuk Kotak / Rahang Tegas)

### 1. Ekstraksi 13 Landmark Wajah
Jurnal mengevaluasi detektor Dlib dan menggunakan 13 titik landmark anatomis kunci:
- Landmark 2 & 14: Jarak tulang pipi / telinga kiri ke kanan (D1)
- Landmark 75 & 79: Lebar dahi kiri ke kanan (D2)
- Landmark 69/72 & 8: Garis rambut puncak kepala ke ujung dagu (D3)
- Landmark 8 & 12: Panjang garis rahang kanan (D4)
- Landmark 4 & 12: Lebar antar sudut rahang bawah (D5)
- Landmark 6 & 10: Lebar rahang bawah area mulut (D6)
- Landmark 7 & 9: Lebar dasar dagu (D7)

*Catatan Implementasi:* Di proyek ini, titik-titik tersebut dipetakan ke **MediaPipe Face Mesh** (478 titik 3D) yang sudah mencakup area dahi dan garis rambut secara presisi tanpa memerlukan kompilasi C++ eksternal.

### 2. 7 Jarak dan 10 Rasio Geometri
- **D1 - D7**: Jarak Euclidean antar titik kunci.
- **N1 - N7**: Jarak ternormalisasi terhadap total penjumlahan 7 jarak ($d'_i = d_i / \sum d_j$).
- **R1 - R10**: 10 rasio proporsi wajah (R1: D2/D1, R2: D1/D3, R3: D2/D3, R4: D1/D5, R5: D6/D5, R6: D4/D6, R7: D6/D1, R8: D5/D2, R9: D4/D5, R10: D7/D6).
- **A1 - A3**: 3 sudut ketegasan garis rahang terhadap sumbu vertikal wajah.

### 3. Seleksi 14 Fitur Terbaik (PCA)
Sesuai Gambar 4 dan Halaman 5 pada jurnal, 14 fitur terpenting yang digunakan untuk klasifikasi adalah:
`[R2, N5, R3, A1, N2, N3, N6, A2, R1, R8, R7, N1, R10, R4]`

### 4. Rekomendasi Kacamata & Gaya Rambut
Sesuai aplikasi jurnal dan studi pendukung *Facial Shape-Based Eyeglass Recommendation*, sistem memberikan saran frame kacamata yang cocok serta gaya rambut yang ideal berdasarkan bentuk wajah terdeteksi.

---

## Struktur Direktori

```text
coba streamlit/
├── app.py                      # Router utama aplikasi Streamlit
├── config.py                   # Konfigurasi path model dan jurnal referensi
├── requirements.txt            # Dependensi pustaka
├── run.bat                     # Shortcut sekali klik di Windows
├── README.md                   # Dokumentasi proyek
│
├── assets/
│   └── sample_faces/           # Contoh citra wajah uji coba (Kartini, Michael Jordan)
│
├── pages_ui/
│   ├── home.py                 # Halaman Beranda & ringkasan jurnal
│   ├── face_shape_page.py      # Halaman Utama: Klasifikasi Bentuk Wajah & Rekomendasi
│   ├── image_processing.py     # Halaman Pengolahan Citra Digital (Filter, Edge, Threshold)
│   ├── computer_vision.py      # Halaman Computer Vision MediaPipe umum
│   └── webcam_test.py          # Halaman Uji Coba Webcam langsung
│
└── utils/
    ├── cv_helpers.py           # Helper transformasi OpenCV & plot histogram
    ├── model_loader.py         # Loader MediaPipe Tasks
    └── face_shape_analyzer.py  # Ekstraktor 13 landmark, 14 fitur PCA, dan klasifikasi jurnal
```

---

## Cara Menjalankan Aplikasi

Klik dua kali file `run.bat` di direktori ini, atau jalankan perintah:

```powershell
py -m streamlit run app.py
```

Aplikasi akan terbuka pada peramban web di alamat: `http://localhost:8501`.
Pilih menu **Klasifikasi Bentuk Wajah (Jurnal)** untuk langsung menguji citra dengan metodologi jurnal.
