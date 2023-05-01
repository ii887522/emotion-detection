import flask
from flask import Flask, request
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np


app = Flask(__name__, static_url_path="", static_folder="static", template_folder="static")


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/detect_emotions", methods=["POST"])
def detect_emotions():
    # Input
    req = json.loads(request.data.decode("utf8"))
    data = req["data"]

    # Convert data to pixels
    image = np.array(Image.open(BytesIO(base64.b64decode(data + "=="))))
    print(image)

    return {"message": "TODO: detect_emotions"}
