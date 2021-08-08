from flask import Flask, render_template, Response, request, url_for
import os
from services import FR_Services
from deepface import DeepFace

app = Flask(__name__)
ser = FR_Services()
# app.config['Upload_Images'] = Upload
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.files:
            if (request.files['unknown_image'] and not request.files['new_image']):
                unknown_image = request.files['unknown_image']
                print(unknown_image.filename)
                unknown_image.save(os.path.join('upload_Images', unknown_image.filename))
                label, dist = ser.face_recognize(unknown_image.filename)
                return render_template('index.html', label=label, dist=dist)
            elif (not request.files['unknown_image'] and request.files['new_image']):
                new_image = request.files['new_image']
                value = request.form['new_label']
                value = value.title()
                reply = ser.retrain(new_image.filename, value)
                return render_template('index.html', reply=reply)
    return render_template('index.html')






if __name__ == '__main__':
    app.run(debug=True)
