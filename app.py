import flask
from flask import Flask


app = Flask(__name__, static_url_path="", static_folder="static", template_folder="static")


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/detect_emotions")
def detect_emotions():
    return ""
