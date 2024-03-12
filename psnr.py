import numpy as np
import cv2
from math import log10, sqrt


def calculate_psnr(original, noisy):
    original = original.astype(np.float32)
    noisy = noisy.astype(np.float32)
    mse = np.mean((original - noisy) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
