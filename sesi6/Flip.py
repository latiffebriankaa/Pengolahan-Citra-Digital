import cv2
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("citra/gunung.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Horizontal flip (dibalik kiri-kanan)
flip_horizontal = cv2.flip(img, 1)

# Vertical flip (dibalik atas-bawah)
flip_vertikal = cv2.flip(img, 0)

# Menampilkan hasil
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(flip_horizontal)
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(flip_vertikal)
plt.title("Vertical Flip")
plt.axis("off")

plt.tight_layout()
plt.show()