from flask import Flask, render_template, Response, request, url_for
import os
import  numpy as np
# from services import FR_Services
import timeit
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from deepface.basemodels import Facenet512
from deepface import DeepFace
import faiss


graph = tf.get_default_graph()
train_emb = np.load('weights/train_emb.npy')
(dim_0, dim_1) = np.shape(train_emb)
train_names = np.load('weights/train_names.npy')
index =  faiss.IndexFlatL2(dim_1)
index.add(train_emb)
print(index.ntotal)

app = Flask(__name__)
# ser = FR_Services()


sess = tf.Session()
set_session(sess)
my_model = Facenet512.loadModel()


def face_recognize(test_image):
    global graph
    global sess
    print('Model Loaded.............')
    path = f'upload_Images/{test_image}'
    t_start = timeit.default_timer()
    with graph.as_default():
        set_session(sess)
        tst_emb = DeepFace.represent(path, model=my_model ,detector_backend='mtcnn')
    t_end = timeit.default_timer()
    print(f'Model Loading Time= {t_end-t_start}')
    print('Embedding Generated.............')
    tst_emb = np.array(tst_emb, dtype=np.float32)
    print('DType Converted.............')
    tst_emb = np.reshape(tst_emb, (1, dim_1))
    print('Array Reshaped.............')
    os.remove(path)
    if not path:
        print('Image File Deleted.............')
    # print(self.train_emb.dtype)
    D, I = index.search(tst_emb, 3)
    print(I)
    idx = I[0][0]
    distance = D[0][0]
    label = train_names[idx]
    return label, distance


# app.config['Upload_Images'] = Upload
@app.route('/', methods=['GET', 'POST'])
def home():
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
                # reply = ser.retrain(new_image.filename, value)
                return render_template('index.html')# , reply=reply
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
