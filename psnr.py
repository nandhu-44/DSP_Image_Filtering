import numpy as np

def calculate_psnr(original, noisy):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images."""
    mse = np.mean((original - noisy) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr
