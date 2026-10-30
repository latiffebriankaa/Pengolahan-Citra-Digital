from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image):
    histogram, _ = np.histogram(
        image.flatten(),
        bins=256,
        range=(0, 256),
    )
    cdf = histogram.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    if cdf_max == cdf_min:
        return image.copy()

    cdf_normalized = (cdf_masked - cdf_min).astype(float) * 255
    cdf_normalized /= cdf_max - cdf_min
    lookup_table = np.ma.filled(cdf_normalized, 0).astype(np.uint8)

    return lookup_table[image]


def to_grayscale(image):
    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        image = image[:, :, :3]

    grayscale = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )
    return grayscale.astype(np.uint8)


image_path = Path(__file__).resolve().parent / "gunung.jpg"
image = to_grayscale(plt.imread(image_path))
result = histogram_equalization(image)

hist_image, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
hist_result, _ = np.histogram(result.flatten(), bins=256, range=(0, 256))

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Gambar Gunung Asli")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Setelah Histogram Equalization")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.plot(hist_image, color="steelblue")
plt.title("Histogram Gambar Asli")
plt.xlabel("Intensitas")
plt.ylabel("Jumlah Piksel")

plt.subplot(2, 2, 4)
plt.plot(hist_result, color="darkorange")
plt.title("Histogram Setelah Equalization")
plt.xlabel("Intensitas")
plt.ylabel("Jumlah Piksel")

plt.tight_layout()
plt.show()