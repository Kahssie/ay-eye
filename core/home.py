from flask import render_template, send_from_directory, request
import os
from core import app
import requests

@app.route("/")
def home():
    return render_template("home.html")


UPLOAD_FOLDER = os.path.join(os.getcwd(),'outputs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# route to hear individual generated clips
@app.route('/uploads/<name>')
def return_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

# route to see the list of generated clips
@app.route('/uploads')
def file_list():
    list = []
    for i in os.listdir(UPLOAD_FOLDER):
        list.append(i)
    return render_template('uploads.html', file_list = list)

@app.route("/forward", methods='POST')
def move_forward():
    #Moving forward code
    if request.method=="POST":
        print("Generating music triggered")
        url = "http://127.0.0.1:5000/generate"
        x = requests.post(url)
        print(x.text)
        return render_template('home.html')