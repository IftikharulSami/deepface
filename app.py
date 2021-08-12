from flask import Flask, render_template, Response, request, url_for
import os
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from services import FR_Services
from deepface import DeepFace
import timeit
from deepface.basemodels import OpenFace

graph = tf.get_default_graph()

app = Flask(__name__)
sess = tf.Session()
set_session(sess)
my_model = OpenFace.loadModel()
ser = FR_Services()

# app.config['Upload_Images'] = Upload
@app.route('/', methods=['GET', 'POST'])
def home():
    global graph
    global sess
    if request.method == 'POST':
        if request.files:
            if (request.files['unknown_image'] and not request.files['new_image']):
                unknown_image = request.files['unknown_image']
                print(unknown_image.filename)
                unknown_image.save(os.path.join('upload_Images', unknown_image.filename))
                t_start = timeit.default_timer()
                with graph.as_default():
                    set_session(sess)
                    tst_emb = DeepFace.represent(f'upload_Images/{test_image}', model=my_model ,detector_backend='mtcnn')
                t_end = timeit.default_timer()
                print(f'Model Loading Time= {t_end-t_start}')
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
