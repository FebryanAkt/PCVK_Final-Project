import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(uploaded_file):
    """Membaca file upload dan mengubahnya menjadi numpy array RGB."""
    image = Image.open(uploaded_file)
    # Konversi ke RGB jika format gambar RGBA atau L
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)

def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """Mengubah citra RGB ke Grayscale."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

def apply_blur(img_rgb: np.ndarray, method: str, ksize: int) -> np.ndarray:
    """Menerapkan filter blurring/smoothing."""
    # Pastikan kernel size ganjil
    if ksize % 2 == 0:
        ksize += 1
    
    if method == "Gaussian Blur":
        return cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)
    elif method == "Median Blur":
        return cv2.medianBlur(img_rgb, ksize)
    elif method == "Bilateral Filter":
        return cv2.bilateralFilter(img_rgb, d=ksize, sigmaColor=75, sigmaSpace=75)
    else:
        return cv2.blur(img_rgb, (ksize, ksize))

def apply_edge_detection(img_rgb: np.ndarray, method: str, th1: int = 100, th2: int = 200) -> np.ndarray:
    """Menerapkan algoritma deteksi tepi."""
    gray = to_grayscale(img_rgb)
    if method == "Canny":
        return cv2.Canny(gray, th1, th2)
    elif method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        return np.uint8(np.clip(magnitude, 0, 255))
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return np.uint8(np.clip(np.absolute(lap), 0, 255))
    return gray

def apply_threshold(img_rgb: np.ndarray, method: str, thresh_val: int = 127) -> np.ndarray:
    """Menerapkan operasi thresholding (binarisasi)."""
    gray = to_grayscale(img_rgb)
    if method == "Binary":
        _, res = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    elif method == "Binary Inverted":
        _, res = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    elif method == "Otsu":
        _, res = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "Adaptive Mean":
        res = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == "Adaptive Gaussian":
        res = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        res = gray
    return res

def convert_color_space(img_rgb: np.ndarray, target_space: str) -> np.ndarray:
    """Konversi ke ruang warna yang berbeda."""
    if target_space == "HSV":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    elif target_space == "LAB":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    elif target_space == "YCrCb":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    return img_rgb

def plot_histogram(img_rgb: np.ndarray):
    """Menghasilkan figure matplotlib histogram warna citra."""
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, alpha=0.8, label=col.upper())
    ax.set_title("Histogram Warna (RGB)", fontsize=10)
    ax.set_xlabel("Intensitas Piksel", fontsize=8)
    ax.set_ylabel("Frekuensi", fontsize=8)
    ax.set_xlim([0, 256])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
