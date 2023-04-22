import cv2
import os
import numpy as np


def load_grayscale_images(dir_path: str) -> np.ndarray: # (N, 48, 48, 1)
    return np.expand_dims(
        np.subtract(
            np.multiply(
                np.array([cv2.cvtColor(cv2.imread(f"{dir_path}/{image_path}"), cv2.COLOR_BGR2GRAY) for image_path in os.listdir(dir_path)]),
                2. / 255
            ),
            1
        ),
        3
    )
