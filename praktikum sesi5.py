from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = Path(__file__).resolve().parent / "Bagian Belakang Motor yang Buram.png"
img = cv2.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

# Sharpening
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)
sharp = cv2.filter2D(rgb, -1, kernel)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for axis, image, title in zip(
    ax,
    [rgb, sharp],
    ["Asli", "Sharpening"],
):
    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")

plt.tight_layout()
plt.show()

# Noisy
# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# mean = cv2.blur(rgb, (3, 3))
# gauss = cv2.GaussianBlur(rgb, (3, 3), 0)
# median = cv2.medianBlur(rgb, 3)

# fig, ax = plt.subplots(1, 4, figsize=(14, 4))
# for axis, image, title in zip(
#     ax,
#     [rgb, mean, gauss, median],
#     ["Asli", "Mean", "Gaussian", "Median"],
# ):
#     axis.imshow(image)
#     axis.set_title(title)
#     axis.axis("off")

# plt.tight_layout(); plt.show()


gunakan 1 buah yang citra noisy, smoting dengan 3 metode mean median tampilkan intensitas citra awal dan akhir
gunakan 1 buah yang citra sharping, analisis citra mana yang bagus, lalu perbaiki metode tersebut dengan metode sharping
bagaimana