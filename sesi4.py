
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

path = Path(__file__).resolve().parent / "gunung.jpg"
my_image = plt.imread(path)


def brightness(img, factor):
    img_result = img.astype(np.float32) + factor
    img_result = np.clip(img_result, 0, 255)
    return img_result.astype(np.uint8)


def adjust_contrast(img, factor):
    mean = 128
    img_result = img.astype(np.float32)
    img_result = mean + factor * (img_result - mean)
    img_result = np.clip(img_result, 0, 255)
    return img_result.astype(np.uint8)


if my_image.ndim == 2:
    display_image = my_image
elif my_image.shape[2] == 4:
    display_image = my_image[:, :, :3]
else:
    display_image = my_image

bright = brightness(display_image, 50)
contrast_image = adjust_contrast(display_image, 2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(display_image, cmap="gray" if display_image.ndim == 2 else None)
plt.title("Gambar Asli")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(bright, cmap="gray" if bright.ndim == 2 else None)
plt.title("Brightness +50")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(
    contrast_image,
    cmap="gray" if contrast_image.ndim == 2 else None,
)
plt.title("Kontras x2")
plt.axis("off")

plt.tight_layout()
plt.show()