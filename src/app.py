import flask
from flask import Flask, request
import json
import base64
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import constants
import tensorflow as tf
import cv2
import converter
from flask_socketio import SocketIO, emit


app = Flask(__name__, static_url_path="", static_folder="../static", template_folder="../static")
socketio = SocketIO(app)
model: tf.keras.Sequential = tf.keras.models.load_model(constants.BEST_MODEL_DIR_PATH)
face_classifier = cv2.CascadeClassifier(constants.HAARCASCADE_FRONTALFACE_DEFAULT_FILE_PATH)


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/detect_emotions", methods=["POST"])
def detect_emotions():
    # Input
    req = json.loads(request.data.decode("utf-8"))
    data = req["data"]

    # Convert data to image
    image = Image.open(BytesIO(base64.b64decode(data[data.index(",") + 1:])))

    # Ensure the image is in RGBA format
    image = image.convert("RGBA")

    # Grayscale the image
    grayscaled_image = ImageOps.grayscale(image)

    # Convert to Numpy arrays
    image = np.array(image)
    grayscaled_image = np.array(grayscaled_image)
    print("Image shape: ", image.shape)
    print("Grayscaled image shape: ", grayscaled_image.shape)

    min_image_dim = min([image.shape[0], image.shape[1]])

    # Detect faces in the image
    for (x, y, w, h) in face_classifier.detectMultiScale(
        grayscaled_image,
        minNeighbors=9,
        minSize=(min_image_dim // 24, min_image_dim // 24),
    ):
        # Crop the region of interest from the grayscaled image
        roi_gray = grayscaled_image[y:y + h, x:x + w]

        # Resize the grayscaled region of interest
        roi_gray = cv2.resize(roi_gray, constants.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # model.predict() expects an array of images instead of 1 image
        roi = np.expand_dims(roi_gray, (0, 3))

        # Make a prediction on the region of interest, then lookup the class
        # [0] because model.predict() returns an array of predictions. We only need the first prediction
        pred = model(roi, training=False).numpy()[0]
        print("Prediction: ", pred)
        label = constants.LABELS[pred.argmax()]
        print("Label: ", label)

        if label != "NF":
            # Indicate a region of interest in the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0, 255), max(round(min_image_dim / 256), 1))

            # Label the image with the predicted emotion
            cv2.putText(
                image,
                label,
                (x, y),
                cv2.FONT_HERSHEY_TRIPLEX,
                max(round(min_image_dim / 512), 1), (255, 0, 0, 255), max(round(min_image_dim / 256), 1)
            )

    # Output
    return {"result": converter.b64encode_image(image)}


@socketio.on("frame")
def on_frame(req):
    # Input
    data = req["data"]

    # Convert data to image
    image = Image.open(BytesIO(base64.b64decode(data[data.index(",") + 1:])))

    # Ensure the image is in RGBA format
    image = image.convert("RGBA")

    # Grayscale the image
    grayscaled_image = ImageOps.grayscale(image)

    # Convert to Numpy arrays
    overlay = np.zeros(np.array(image).shape, dtype=np.uint8)
    grayscaled_image = np.array(grayscaled_image)
    print("Overlay shape: ", overlay.shape)
    print("Grayscaled image shape: ", grayscaled_image.shape)

    min_image_dim = min([overlay.shape[0], overlay.shape[1]])

    # Detect faces in the image
    for (x, y, w, h) in face_classifier.detectMultiScale(
        grayscaled_image,
        minNeighbors=9,
        minSize=(min_image_dim // 24, min_image_dim // 24),
    ):
        # Crop the region of interest from the grayscaled image
        roi_gray = grayscaled_image[y:y + h, x:x + w]

        # Resize the grayscaled region of interest
        roi_gray = cv2.resize(roi_gray, constants.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # model.predict() expects an array of images instead of 1 image
        roi = np.expand_dims(roi_gray, (0, 3))

        # Make a prediction on the region of interest, then lookup the class
        # [0] because model.predict() returns an array of predictions. We only need the first prediction
        pred = model(roi, training=False).numpy()[0]
        print("Prediction: ", pred)
        label = constants.LABELS[pred.argmax()]
        print("Label: ", label)

        if label != "NF":
            # Indicate a region of interest in the image
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0, 255), max(round(min_image_dim / 256), 1))

            # Label the image with the predicted emotion
            cv2.putText(
                overlay,
                label,
                (x, y),
                cv2.FONT_HERSHEY_TRIPLEX,
                max(round(min_image_dim / 512), 1), (255, 0, 0, 255), max(round(min_image_dim / 256), 1)
            )

    # Output
    emit("overlay", {"data": converter.b64encode_image(overlay)})


if __name__ == '__main__':
    socketio.run(app)
