import numpy as np
import cv2
def deblur(image, kernel_size = 5, snr = 10):
    # Convert image to grayscale
    # Add Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Wiener filter
    wiener_filter = np.fft.fft2(blurred) / (np.fft.fft2(blurred) + 1/snr)
    deblurred = np.fft.ifft2(np.fft.fft2(blurred) * wiener_filter).real

    return deblurred