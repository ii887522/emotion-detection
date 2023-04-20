import tkinter as tk
import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import neurolab as nl


INPUT_DIR_PATH = "res/emotions"
TARGET_IMAGE_SIZE = (48, 48)


def main():
    # Show a window for presentation
    window = tk.Tk()
    window.title("Emotion Detection")
    window.geometry("360x360")
    window.resizable(False, False)

    # Setup ANN
    net = nl.net.newff(
        [[0, 1]] * (TARGET_IMAGE_SIZE[0] * TARGET_IMAGE_SIZE[1]), [10, 10, 10, 10, 8]
    )

    image_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.5, 1.5],
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.3,
        preprocessing_function=lambda image: np.expand_dims(
            cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE), 2
        )
    )

    # Train loop
    for ((train_xs, train_ys), (test_xs, test_ys)) in zip(image_gen.flow_from_directory(
        directory=INPUT_DIR_PATH,
        target_size=TARGET_IMAGE_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        subset="training"
    ), image_gen.flow_from_directory(
        directory=INPUT_DIR_PATH,
        target_size=TARGET_IMAGE_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation"
    )):
       train_xs = np.reshape(train_xs, (train_xs.shape[0], train_xs.shape[1] * train_xs.shape[2]))
       test_xs = np.reshape(test_xs, (test_xs.shape[0], test_xs.shape[1] * test_xs.shape[2]))
       error = net.train(train_xs, train_ys, epochs=1, show=1)
       print(error)

    window.mainloop()


if __name__ == "__main__":
    main()
