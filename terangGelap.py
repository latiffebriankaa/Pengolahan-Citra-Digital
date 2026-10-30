from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ubah_menjadi_gelap(citra, pengurangan=80):
    """Mengurangi intensitas piksel citra agar tampil lebih gelap."""
    citra_float = citra.astype(np.float32)
    citra_gelap = citra_float - pengurangan
    return np.clip(citra_gelap, 0, 255).astype(np.uint8)


def tampilkan_citra(citra, judul):
    plt.imshow(citra)
    plt.title(judul)
    plt.axis("off")


lokasi_gambar = Path(__file__).resolve().parent / "platnomor.jpeg"
citra_asli = plt.imread(lokasi_gambar)

# Jika format pembacaan menghasilkan float 0-1, ubah ke rentang 0-255.
if np.issubdtype(citra_asli.dtype, np.floating) and citra_asli.max() <= 1:
    citra_asli = (citra_asli * 255).astype(np.uint8)
else:
    citra_asli = citra_asli.astype(np.uint8)

# Hapus kanal alpha jika gambar memilikinya.
if citra_asli.ndim == 3 and citra_asli.shape[2] == 4:
    citra_asli = citra_asli[:, :, :3]

citra_gelap = ubah_menjadi_gelap(citra_asli, pengurangan=80)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
tampilkan_citra(citra_asli, "Citra Plat Nomor Asli")

plt.subplot(1, 2, 2)
tampilkan_citra(citra_gelap, "Citra Plat Nomor Lebih Gelap")

plt.tight_layout()
plt.show()