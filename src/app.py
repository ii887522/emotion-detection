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


app = Flask(__name__, static_url_path="", static_folder="../static", template_folder="../static")
model: tf.keras.Sequential = tf.keras.models.load_model(constants.BEST_MODEL_DIR_PATH)
face_classifier = cv2.CascadeClassifier(constants.HAARCASCADE_FRONTALFACE_DEFAULT_FILE_PATH)


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/detect_emotions", methods=["POST"])
def detect_emotions():
    # Input
    req = json.loads(request.data.decode("utf8"))
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

    # Detect faces in the image
    for (x, y, w, h) in face_classifier.detectMultiScale(grayscaled_image):
        # Indicate a region of interest in the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the region of interest from the grayscaled image
        roi_gray = grayscaled_image[y:y + h, x:x + w]

        # Resize the grayscaled region of interest
        roi_gray = cv2.resize(roi_gray, constants.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Normalize the grayscaled region of interest
            roi = roi_gray.astype("float") / 255.0

            # model.predict() expects an array of images instead of 1 image
            roi = np.expand_dims(roi, 0)

            # Make a prediction on the region of interest, then lookup the class
            # [0] because model.predict() returns an array of predictions. We only need the first prediction
            preds = model.predict(roi)[0]
            print("Prediction: ", preds)
            label = constants.LABELS[preds.argmax()]
            print("Label: ", label)

            # Label the image with the predicted emotion
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)

        else:
            # Display error message on the image
            cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)

    # Convert the image into base64 encoded string
    return {"result": base64.b64encode(image.tobytes())}
