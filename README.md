# Bridges Eyewear - AI Smart Fit & Vision Studio
Aplikasi Pengolahan Citra Digital dan Visi Komputer yang menggabungkan estetika kontemporer **Bridges Eyewear** (Optik Melawai Group) dengan metodologi ilmiah:
**"Human Face Shape Classification with Machine Learning"** (Ashwinee Mehta & Taha Mohamed Mahmoud).

---

## 🌟 Fitur Utama & Desain Ala Bridges Eyewear

1. **Editorial Brand Identity (Bridges Eyewear)**
   - Top announcement ticker ala web resmi [bridgeseyewear.com](https://www.bridgeseyewear.com/id/).
   - Tipografi mewah modern (Plus Jakarta Sans, Playfair Display serif headings, Space Mono).
   - Palet warna hitam, putih bersih, hangat bronze champagne gold (`#9D784C`), dan adaptasi otomatis Light/Dark mode.
   - Footer resmi dengan tautan gerai, garansi 1 tahun Optik Melawai, dan katalog.

2. **AI Face Scan & Smart Fit Sizing**
   - Pemindaian 81 Landmark Dlib (Codeniko model) untuk mendeteksi 5 bentuk wajah (*Heart, Oblong, Oval, Round, Square*).
   - Pengukuran antropometri presisi: Lebar Pelipis (D2), Tulang Pipi (D1), Jembatan Hidung, dan Lebar Rahang (D5).
   - Kalkulasi kode optik internasional (contoh: `50 [] 18 - 145 mm`).
   - Pencocokan otomatis langsung dengan koleksi kacamata Bridges Eyewear yang paling pas.

3. **Koleksi & Katalog Frame Kacamata Bridges**
   - Katalog interaktif frame optik dan sunwear (Full Italian Acetate, Bridges+ Titanium 11g, Combination Frame, Active TR-90).
   - Filter real-time berdasarkan bentuk wajah, siluet frame, material, dan harga.

4. **Laboratorium Pengolahan Citra Digital (PCD Studio)**
   - Binarisasi adaptif (Otsu, Adaptive Gaussian, Thresholding).
   - Penapisan spasial / blurring (Gaussian, Median, Bilateral).
   - Deteksi tepi rim frame kacamata (Canny, Sobel, Laplacian).
   - Ruang warna (RGB, HSV, LAB, YCrCb) dan analisis histogram.

5. **Computer Vision & Live Webcam Fitting**
   - Inferensi MediaPipe 478 koordinat 3D wajah dan 21 sendi tangan.
   - Deteksi wajah Viola-Jones (Haar Cascade).
   - Live Webcam Virtual Fit: ambil foto langsung dari webcam laptop untuk diagnosis ukuran frame instan.

---

## 🚀 Cara Menjalankan

1. Klik ganda pada berkas **`run.bat`**, atau
2. Jalankan perintah terminal:
   ```bash
   streamlit run app.py
   ```
