from flask import Flask, render_template, request, jsonify
import timeit
from mtcnn import MTCNN
from deepface import DeepFace
from deepface.basemodels import Facenet512
import faiss
import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
# from numpy import matlib as mb

app = Flask(__name__)
emb_dim = 512  # 128 for Facenet and 512 for Facenet512
detector = MTCNN()
my_model = Facenet512.loadModel('model_weights/facenet512_weights.h5')
train_emb = np.load('weights/train_emb.npy')
train_names = np.load('weights/train_names.npy')
index = faiss.IndexFlatL2(emb_dim)
index.add(train_emb)
# print(f'Shape of Train Emb {np.shape(train_emb)}')

def gen_threads(detected_faces, n_faces, frame, frameNo):
    start_time_thread = timeit.default_timer()
    for i in range(n_faces):
        face = detected_faces[i]['box']
        (x, y, w, h) = face
        image = frame[y:y + h, x:x + w]
        globals()['t%s' % i] = i
        globals()['t%s' % i] = threading.Thread(target=compare_dims(face, globals()['t%s' % i], detected_faces, frameNo, image))
        globals()['t%s' % i].start()
    end_time_thread = timeit.default_timer()
    # print(f'Total Time for Thread Generation is {end_time_thread - start_time_thread}')

def compare_dims(object, t_num, detected_faces, frameNo, image):
    (x_nose, y_nose) = detected_faces[0]['keypoints']['nose']
    # print(f'X of Nose in Orig Image {x_nose}, Y of Nose in Orig Image {y_nose}')
    start_time_comp = timeit.default_timer()
    # cv2.imshow('Face',face)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    # print(f'Thread Number is {t_num + 1}')
    (x, y, w, h) = object
    # print(x,y,w,h)
    h_limit = 112
    w_limit = 112
    if (h >= h_limit and w >= w_limit):
        # print(f'Height = {h} and Width = {w}')
        check_orientation(detected_faces, frameNo, image)
    end_time_comp = timeit.default_timer()
    # print(f'Total Time for dims comparison is {end_time_comp - start_time_comp}')
    return

def check_orientation(detected_face, frameNo, image):
    # (h, w, c) = np.shape(face)
    # det_face = detector.detect_faces(face)
    # if len(det_face)>0:
    (x, y, width, height) = detected_face[0]['box']
    # print(f'Width of single face {width}')
    # print(f'Width of detected face {width} and Height of detected face {height}')
    mid_width = int(width/2)
    mid_height = int(height / 2)
    x_disp = int(width * .15)
    y_disp = int(height * .15)
    # print(x_disp, y_disp)
    # print(f'Mid of Width {mid_width} Mid of Height {mid_height}')
    (x_nose_orig, y_nose_orig) = detected_face[0]['keypoints']['nose']
    x_nose =  x_nose_orig - x
    y_nose = y_nose_orig - y
    # print(f'Nose Cordinates {x_nose}, {y_nose}')

    if (x_nose < mid_width-x_disp or x_nose > mid_width+x_disp) or (y_nose < mid_height-y_disp or y_nose > mid_height+y_disp):
        print(f'Frame No {frameNo} - Profile Face')
    else:
        print(f'Frame No {frameNo} - Fontal Face')
        recognition(image, frameNo)
    # else:
    #     print('No face Detected')

def recognition(face, frameNo):
    tst_emb =  DeepFace.represent(face, model=my_model, detector_backend='mtcnn')
    tst_emb = np.array(tst_emb, dtype=np.float32)
    tst_emb = np.reshape(tst_emb, (1, emb_dim))
    print(np.shape(tst_emb))
    D, I = index.search(tst_emb, 3)
    idx = I[0][0]
    distance = D[0][0]
    label = train_names[idx]
    print(f'Person Identified as {label} in frame {frameNo}')

def attendance():
    pass

def model_train():
    pass

def test(): # Live Stream
    cap = cv2.VideoCapture(0)
    frameNo = 0
    while cap:
        frameNo += 1
        print(f'Frame Counter = {frameNo}')
        start_frame = timeit.default_timer()
        ret, frame = cap.read()
        # cv2.imshow('Face',frame)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        # (he, wi, ch) = frame.shape
        # print(f'Height is {he}, Width is {wi}')
        # end_frame = timeit.default_timer()
        # start_gray = timeit.default_timer()
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # gray = np.expand_dims(gray, axis=1)
        # print(gray)
        # end_gray = timeit.default_timer()
        # print(f'Framing Time = {end_gray-start_gray}')
        start = timeit.default_timer()
        if frameNo % 3 == 0:
            detected_face = detector.detect_faces(frame)
            end = timeit.default_timer()
            # check_orientation(detected_face)
            print(f'Face detection Time = {end - start_frame}')
            num_faces = len(detected_face)
            print(f'Faces = {num_faces}')
            if num_faces == 0:
                continue
            gen_threads(detected_face, num_faces, frame, frameNo)
            finsih = timeit.default_timer()
            print(f"Total Time is {finsih - start_frame}")
        else:
            continue
        # break


def test1(): # Gallery Image
    img = cv2.imread('/home/ncbc-iftikhar/Downloads/sample.jpg')
    start = timeit.default_timer()
    detected_face = detector.detect_faces(img)
    end = timeit.default_timer()
    print(f'Face detection Time = {end - start}')
    print(len(detected_face))
    # if faces == 0:
    #     continue
    # check_orientation(detected_face)
    # for i in range(faces):
    #     face = detected_face[i]['box']
    #     globals()['t%s' % i] = i
    #     globals()['t%s' % i] = threading.Thread(target=compare_dims(face, globals()['t%s' % i]))
    #     globals()['t%s' % i].start()

test()
if __name__ == '__main__':
    app.run(debug=True)