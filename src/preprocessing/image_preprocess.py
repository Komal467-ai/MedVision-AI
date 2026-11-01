import cv2
import numpy as np


def preprocess_image(image, target_size=(224, 224)):
    """Preprocessing pipeline for medical images."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    resized = cv2.resize(enhanced, target_size)
    normalized = resized.astype(np.float32) / 255.0
    three_channel = np.stack([normalized] * 3, axis=-1)

    return three_channel, enhanced