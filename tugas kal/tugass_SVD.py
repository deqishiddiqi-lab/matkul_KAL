import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load gambar (RGB, bukan grayscale)
img = Image.open("sakera.png")
img_array = np.array(img)

# Pisahkan channel warna
R = img_array[:, :, 0]
G = img_array[:, :, 1]
B = img_array[:, :, 2]

# Fungsi SVD per channel
def svd_channel(channel, k):
    U, S, VT = np.linalg.svd(channel, full_matrices=False)
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))

# Nilai kompresi
k = 200

# Rekonstruksi tiap channel
R_svd = svd_channel(R, k)
G_svd = svd_channel(G, k)
B_svd = svd_channel(B, k)

# Gabungkan kembali
img_svd = np.stack((R_svd, G_svd, B_svd), axis=2)

# Clip nilai agar valid (0-255)
img_svd = np.clip(img_svd, 0, 255).astype(np.uint8)

# Tampilkan
plt.imshow(img_svd)
plt.title(f"Gambar Berwarna (k={k})")
plt.axis('off')
plt.show()