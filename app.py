import flask
from flask import Flask


app = Flask(__name__, static_url_path="", static_folder="static", template_folder="static")


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/detect_emotions", methods=["POST"])
def detect_emotions():
    return {"message": "TODO: detect_emotions"}
