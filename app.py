import io
from flask import Flask, render_template, request, jsonify
import os
import base64
import  numpy as np
import timeit
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from deepface.basemodels import Facenet512
from deepface import DeepFace
import faiss
from PIL import Image

#-------------------------------------
graph = tf.get_default_graph()

#-------------------------------------
app = Flask(__name__)
# ser = FR_Services()

sess = tf.Session()
set_session(sess)
my_model = Facenet512.loadModel()

def loadEmb():
    train_emb = np.load('weights/train_emb.npy')
    (dim_0, dim_1) = np.shape(train_emb)
    train_names = np.load('weights/train_names.npy')
    index = faiss.IndexFlatL2(dim_1)
    index.add(train_emb)
    # print(index.ntotal)
    return index, dim_0, dim_1, train_names

def face_recognize(test_image_path):
    global graph
    global sess
    (index, dim_0, dim_1, train_names) = loadEmb()
    path = test_image_path
    t_start = timeit.default_timer()
    with graph.as_default():
        set_session(sess)
        tst_emb = DeepFace.represent(path, model=my_model ,detector_backend='mtcnn')
    t_end = timeit.default_timer()
    print(f'Model Loading Time= {t_end-t_start}')
    # print('Embedding Generated.............')
    tst_emb = np.array(tst_emb, dtype=np.float32)
    # print('DType Converted.............')
    tst_emb = np.reshape(tst_emb, (1, dim_1))
    # print('Array Reshaped.............')
    os.remove(path)
    D, I = index.search(tst_emb, 3)
    # print(I)
    idx = I[0][0]
    label = 'Unknown Face'
    distance = D[0][0] / 10
    if distance <= 45.0:
        label = train_names[idx]
    return label, distance

def b64toImg(enc, name, ext):
    image = base64.b64decode(enc)
    filename = f'{name}.{ext}'
    imgPath = f'temp/{filename}'
    img = Image.open(io.BytesIO(image))
    img.save(imgPath, ext)
    return imgPath

def updateEmd(imgpath, label):
    global count
    names = []
    (index, dim_0, dim_1, train_names) = loadEmb()
    print(f'before Update - {index.ntotal}')
    newImg = Image.open(imgpath)
    Train_Images = 'Train_images'
    for name in os.listdir(Train_Images):
        names.append(name)
    if label not in names:
        os.mkdir(f'{Train_Images}/{label}')
        newImg.save(f'{Train_Images}/{label}/{label}.png', 'png')
    else:
        print(count)
        newImg.save(f'{Train_Images}/{label}/IM{count}.png', 'png')
        count = count+1
    os.remove(imgpath)
    train_emb = []
    train_names = []
    for lbl in os.listdir(Train_Images):
        for img_file in os.listdir(f'{Train_Images}/{lbl}'):
            with graph.as_default():
                set_session(sess)
                emb = DeepFace.represent(f'{Train_Images}/{lbl}/{img_file}', model=my_model, detector_backend='mtcnn')
                train_emb.append(emb)
                train_names.append(lbl)
    train_emb = np.array(train_emb, dtype=np.float32)
    # print(np.shape(train_emb))
    np.save('weights/train_emb.npy', train_emb)
    np.save('weights/train_names.npy', train_names)
    return 'Successfully added.'

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('welcome.html')
#
@app.route('/login', methods=['GET', 'POST'])
def login():
    usr = ['admin', 'atif', 'imran', 'husain', 'iftikhar']
    pwd = ['admin', 'atif', 'imran', 'husian', 'iftikhar']
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return render_template('welcome.html')
    return render_template('login.html', error=error)

@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        if request.form['submit']=='Saved Image':
            return render_template('RecFromFile.html')
        elif request.form['submit']=='Live Stream':
            return render_template('RecFromCamera.html')
        elif request.form['submit']=='Add Image':
            return render_template('NewUser.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        unknown_image = request.files['unknown_image']
        unknown_image.save(os.path.join('upload_Images', unknown_image.filename))
        imgPath = f'upload_Images/{unknown_image.filename}'
        st_time = timeit.default_timer()
        label, dist = face_recognize(imgPath)
        end_time = timeit.default_timer()
        print(label)
        print('Elapsed Time: ', end_time - st_time)
        return render_template('RecFromFile.html', label=label, dist=dist)
    return render_template('recFromFile.html')

@app.route('/recognizefromcamera', methods=['GET', 'POST'])
def recognizefromcamera():
    if request.method == 'POST':
        imgEnc = request.json['imageBase64']
        # imgEnc = request.values['imageBase64']
        (data, enc) = imgEnc.split(';')
        (type, ext) = data.split('/')
        (_, encod) = enc.split(',')
        name = 'Temp'
        imgPath = b64toImg(encod, name, ext)
        st_time = timeit.default_timer()
        label, dist = face_recognize(imgPath)
        end_time = timeit.default_timer()
        print(label)
        print('Elapsed Time: ', end_time - st_time)
        return jsonify({'label': label, 'dist': dist})
    return render_template('RecFromCamera.html')

@app.route('/loadDiv', methods=['GET', 'POST'])
def loadDiv():
    if request.method == 'POST':
        option =  request.form.getlist('options')
        if not option:
            return render_template('NewUser.html')
        elif option[0] == 'camera':
            return render_template('camera.html')
        elif option[0] == 'gallery':
            return render_template('gallery.html')
    return render_template('NewUser.html')

@app.route('/addimagefromgallery', methods=['GET', 'POST'])
def addimagefromgallery():
    if request.method == 'POST':
        name = request.json['label']
        imgEnc = request.json['imageBase64']
        (data, enc) = imgEnc.split(';')
        (type, ext) = data.split('/')
        (_, encod) = enc.split(',')
        imgPath = b64toImg(encod, name, ext)
        updateEmd(imgPath, name)
        resp = 'Image added Successfully.'
        return jsonify({'resp': resp})
    return render_template('gallery.html')
@app.route('/addimagefromcamera', methods=['GET', 'POST'])
def addimagefromcamera():
    if request.method == 'POST':
        name = request.json['label']
        imgEnc = request.json['imageBase64']
        (data, enc) = imgEnc.split(';')
        (type, ext) = data.split('/')
        (_, encod) = enc.split(',')
        imgPath = b64toImg(encod, name, ext)
        updateEmd(imgPath, name)
        resp = 'Image added Successfully.'
        return jsonify({'resp': resp})
    return render_template('camera.html')


if __name__ == '__main__':
    count = 1
    app.run(debug=True)