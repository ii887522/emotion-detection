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
    face_coords = face_classifier.detectMultiScale(grayscaled_image)

    for (x, y, w, h) in face_coords:
        # Indicate a region of interest in the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = grayscaled_image[y:y + h, x:x + w]

        # Resize the grayscaled region of interest
        roi_gray = cv2.resize(roi_gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Normalize the grayscaled region of interest
            roi = roi_gray.astype("float") / 255.0

            roi = tf.keras.preprocessing.image.img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            preds = model.predict(roi)[0]
            label = constants.LABELS[preds.argmax()]

            label_position = (x, y)
            cv2.putText(images[i],label,label_position,cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),3)
        else:
            cv2.putText(images[i],'No Face Found',(20,60),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),3)

    return {"message": "TODO: detect_emotions"}
