import cv2
from PIL import Image, ImageTk
from typing import Tuple


def load_tk_image(file_path: str, size: Tuple[int, int], interpolation: int) -> ImageTk.PhotoImage:
    # Load an image
    image = cv2.imread(file_path)

    # Resize the image
    image = cv2.resize(image, size, interpolation=interpolation)

    # Blur the image to reduce noise in edge detection
    image = cv2.GaussianBlur(image, (7, 7), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE)

    # TODO: Augment the image

    # Convert the image to be presentable in a Tkinter window
    image = Image.fromarray(image)

    return ImageTk.PhotoImage(image)
