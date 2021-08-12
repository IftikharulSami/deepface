from flask import Flask, render_template, Response, request, url_for
import os
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from services import FR_Services
from deepface import DeepFace
import timeit
from deepface.basemodels import OpenFace
import faiss
import numpy as np

graph = tf.get_default_graph()

app = Flask(__name__)
sess = tf.Session()
set_session(sess)
my_model = OpenFace.loadModel()
ser = FR_Services()

train_emb = np.load('weights/train_emb.npy')
(dim_0, dim_1) = np.shape(train_emb)
train_names = np.load('weights/train_names.npy')
index =  faiss.IndexFlatL2(dim_1)
# def load_model():
#     global model
#     model = Facenet512.loadModel('model_weights/facenet512_weights.h5')
def face_recognize(test_image):
    global graph
    global sess
    print('Model Loaded.............')
    t_start = timeit.default_timer()
    with graph.as_default():
        set_session(sess)
        tst_emb = DeepFace.represent(f'upload_Images/{test_image}', model=my_model ,detector_backend='mtcnn')
    t_end = timeit.default_timer()
    print(f'Model Loading Time= {t_end-t_start}')
    print('Embedding Generated.............')
    tst_emb = np.array(tst_emb, dtype=np.float32)
    print('DType Converted.............')
    tst_emb = np.reshape(tst_emb, (1, dim_1))
    print('Array Reshaped.............')
    # os.remove(f'upload_Images/{test_image}')
    print('Image File Deleted.............')
    # print(self.train_emb.dtype)
    index.add(train_emb)
    D, I = index.search(tst_emb, 3)
    print(I)
    idx = I[0][0]
    distance = D[0][0]
    label = train_names[idx]
    return label, distance

# app.config['Upload_Images'] = Upload
@app.route('/', methods=['GET', 'POST'])
def home():
    global graph
    global sess
    if request.method == 'POST':
        if request.files:
            if (request.files['unknown_image'] and not request.files['new_image']):
                unknown_image = request.files['unknown_image']
                unknown_image.save(os.path.join('upload_Images', unknown_image.filename))
                st_time = timeit.default_timer()
                label, dist = face_recognize(unknown_image.filename)
                end_time = timeit.default_timer()
                print('Elapsed Time: ', end_time - st_time)
                return render_template('index.html', label=label, dist=dist)
            elif (not request.files['unknown_image'] and request.files['new_image']):
                new_image = request.files['new_image']
                value = request.form['new_label']
                value = value.title()
#               reply = ser.retrain(new_image.filename, value)
                return render_template('index.html', reply=reply)
    return render_template('index.html')






if __name__ == '__main__':
    app.run(debug=True)
