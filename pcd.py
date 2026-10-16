import cv2
import matplotlib.pyplot as plt

# Membaca citra
img = cv2.imread('gunung.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Konversi BGR ke grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Informasi dimensi citra warna
tinggi, lebar, channel = img.shape
print(f"Dimensi RGB: {lebar} x {tinggi} piksel, {channel} saluran warna.")

# Menampilkan RGB dan grayscale berdampingan
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Citra RGB")
plt.axis("on")

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap="hot") # bisa pake gray, hot, cool, viridis, plasma, inferno, magma, jet
plt.title("Citra Grayscale")
plt.axis("on")

plt.tight_layout()
plt.show()