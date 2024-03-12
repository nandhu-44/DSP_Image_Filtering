import cv2
import numpy as np


def laplacian_filter1(image):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return cv2.filter2D(image, -1, kernel)


def laplacian_filter2(image):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return cv2.filter2D(image, -1, kernel)


def laplacian_filter3(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def laplacian_filter4(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def discontinuity_map(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)
