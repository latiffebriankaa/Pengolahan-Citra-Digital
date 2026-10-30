import cv2
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("citra/gunung.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tinggi, lebar = img.shape[:2]

# Menentukan titik pusat
pusat = (lebar // 2, tinggi // 2)

# Rotasi 45, 90, dan 180 derajat
hasil = {}
for sudut in (45, 90, 180):
    M = cv2.getRotationMatrix2D(pusat, sudut, 1.0)
    hasil[sudut] = cv2.warpAffine(img, M, (lebar, tinggi))

# Menampilkan hasil
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Citra Asli")
plt.axis("off")

for i, sudut in enumerate((45, 90, 180), start=2):
    plt.subplot(2, 2, i)
    plt.imshow(hasil[sudut])
    plt.title(f"Rotasi {sudut} Derajat")
    plt.axis("off")

plt.tight_layout()
plt.show()