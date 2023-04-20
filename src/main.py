import tkinter as tk
import common
import cv2
# from keras.preprocessing.image import ImageDataGenerator

def main():
    # Show a window for presentation
    window = tk.Tk()
    window.title("Emotion Detection")
    window.geometry("360x360")
    window.resizable(False, False)

    # Load an image
    image = common.load_tk_image(
        file_path="res/emotions-1/anger/2Q__ (1)_face.png",
        size=(96, 96),
        interpolation=cv2.INTER_AREA
    )

    # Show the image
    tk.Label(window, image=image).pack()

    window.mainloop()


if __name__ == "__main__":
    main()
