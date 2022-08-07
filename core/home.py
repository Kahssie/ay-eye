from flask import render_template, send_from_directory
import os
from core import app

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