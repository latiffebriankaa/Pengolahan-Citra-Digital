import cv2
import matplotlib.pyplot as plt

img = cv2.imread("citra/gunung.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Scaling faktor 0,5 (diperkecil)
kecil = cv2.resize(
    img,
    None,
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_AREA
)

# Scaling faktor 2 (diperbesar)
besar = cv2.resize(
    img,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_LINEAR
)

# Menampilkan informasi ukuran
print("Ukuran gambar asli:", img.shape)
print("Ukuran setelah scaling 0,5:", kecil.shape)
print("Ukuran setelah scaling 2:", besar.shape)

# Menampilkan gambar
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title(f"Asli: {img.shape[1]} x {img.shape[0]}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(kecil)
plt.title(f"Scaling 0,5: {kecil.shape[1]} x {kecil.shape[0]}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(besar)
plt.title(f"Scaling 2: {besar.shape[1]} x {besar.shape[0]}")
plt.axis("off")

plt.tight_layout()
plt.show()