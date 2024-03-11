import cv2
import numpy as np

def apply_average_filter(image, kernel_size):
    """Apply average filter to the image."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

def apply_gaussian_filter(image, kernel_size):
    """Apply Gaussian filter to the image."""
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel = np.dot(kernel, kernel.T)
    filtered = cv2.filter2D(image, -1, kernel)
    return filtered
