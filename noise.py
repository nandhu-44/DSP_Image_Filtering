import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image."""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)
