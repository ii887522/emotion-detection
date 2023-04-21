import tkinter as tk
import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.regularizers import L2
from keras.optimizers import Adam


INPUT_DIR_PATH = "res/emotions"
TARGET_IMAGE_SIZE = (48, 48)
EMOTION_CLASS_COUNT = 8


def main():
    # Show a window for presentation
    window = tk.Tk()
    window.title("Emotion Detection")
    window.geometry("360x360")
    window.resizable(False, False)

    # Build an artificial neural network
    model = Sequential()
    model.add(
        Reshape(
            (TARGET_IMAGE_SIZE[0] * TARGET_IMAGE_SIZE[1],),
            input_shape=(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 1)
        )
    )
    model.add(
        Dense(
            TARGET_IMAGE_SIZE[0] * TARGET_IMAGE_SIZE[1] + EMOTION_CLASS_COUNT,
            "tanh",
            kernel_regularizer=L2(0.001),
        )
    )
    model.add(
        Dense(
            EMOTION_CLASS_COUNT,
            "sigmoid",
            kernel_regularizer=L2(0.001),
        )
    )
    model.summary()
    model.compile(Adam(0.01), "categorical_crossentropy", ["accuracy"], jit_compile=True)

    # Prepare datasets
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
    train_gen = image_gen.flow_from_directory(
        directory=INPUT_DIR_PATH,
        target_size=TARGET_IMAGE_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        subset="training"
    )
    valid_gen = image_gen.flow_from_directory(
        directory=INPUT_DIR_PATH,
        target_size=TARGET_IMAGE_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation"
    )

    # Train the network
    model.fit(train_gen, epochs=50, validation_data=valid_gen, steps_per_epoch=2310, validation_steps=989)

    window.mainloop()


if __name__ == "__main__":
    main()
