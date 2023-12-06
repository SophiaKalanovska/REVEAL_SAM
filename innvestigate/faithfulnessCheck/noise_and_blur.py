import numpy as np
import cv2

def add_gaussian_noise(image, mean=0, sigma=0.1):
    """
    Adds Gaussian noise to an image.

    :param image: A NumPy array representing the image.
    :param mean: Mean of the Gaussian noise.
    :param sigma: Standard deviation of the Gaussian noise.
    :return: Image with Gaussian noise added.
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are within proper range
    return noisy_image.astype(np.uint8)




def gaussian_blur(image, kernel_size=(5, 5)):
    """
    Applies Gaussian blur to an image.

    :param image: A NumPy array representing the image.
    :param kernel_size: Size of the Gaussian kernel.
    :return: Blurred image.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)