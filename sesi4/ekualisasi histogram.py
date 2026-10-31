from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy import ndimage


def get_histogram(image):
    return ndimage.histogram(image, 0, 255, 256)

def histogram_equalization(image):
    histogram = get_histogram(image)
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


image_path = Path(__file__).resolve().parent.parent / "citra" / "platnomor.jpeg"
if not image_path.is_file():
    raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

image = to_grayscale(iio.imread(image_path))
result = histogram_equalization(image)

hist_image = get_histogram(image)
hist_result = get_histogram(result)

print("Statistik tingkat keabuan")
for nama, citra in (("Citra asli", image), ("Setelah equalization", result)):
    print(
        f"{nama:<22} min={citra.min():3d}  max={citra.max():3d}  "
        f"rata-rata={citra.mean():6.2f}  std={citra.std():6.2f}"
    )

iio.imwrite(Path(__file__).resolve().parent / "hasil_ekualisasi.png", result)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Gambar Asli (Grayscale)")
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
plt.savefig(Path(__file__).resolve().parent / "perbandingan_hasil.png", dpi=130)
plt.show()