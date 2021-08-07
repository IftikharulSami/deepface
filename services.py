import faiss
from deepface import DeepFace
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
class FR_Services ():
    def __init__(self):
        self.train_emb = np.load('weights/train_emb.npy')
        (self.dim_0, self.dim_1) = np.shape(self.train_emb)
        self.train_names = np.load('weights/train_names.npy')
        self.index =  faiss.IndexFlatL2(self.dim_1)


    def face_recognize(self, test_image):
        tst_emb = DeepFace.represent(f'upload_Images/{test_image}', model_name='Facenet512', detector_backend='mtcnn')
        tst_emb = np.array(tst_emb, dtype=np.float32)
        tst_emb = np.reshape(tst_emb, (1, self.dim_1))
        os.unlink(f'upload_Images/{test_image}')
        print(self.train_emb.dtype)
        self.index.add(self.train_emb)
        D, I = self.index.search(tst_emb, 3)
        idx = I[0][0]
        distance = D[0][0]
        label = self.train_names[idx]
        return label, distance

    # def gen_labels_from_images(self):
    #     pass
    #
    # def get_image_from_file(address):
    #     img = cv2.imread(address)
    #     FR_Services.detect_face_locations(img)
    # def read_image(image):
        # img = fr.load_image_file(image)
        # return FR_Services.detect_face_locations(img)
    # def update_enc_labels(self, image, label):
        # print(np.shape(self.train_names))
        # d = r'encoding/'
        # filesToRemove = [os.path.join(d, f) for f in os.listdir(d)]
        # for f in filesToRemove:
        #     os.remove(f)
        # new_enc = []
        # Train_images = r'/home/ncbc-iftikhar/Facial-Recognition/images/Train'
        # for name in os.listdir(Train_images):
        #     for filename in os.listdir(f'{Train_images}/{name}'):
        #         image = fr.load_image_file(f'{Train_images}/{name}/{filename}')
        #         locations = fr.face_locations(image, model='cnn')  # cnn
        #         encoding = fr.face_encodings(image, locations)[0]
        #         train_faces.append(encoding)
        #         train_names.append(name)
        # np.save(r'/encoding/encoding/encoding.npy', train_faces)
        # np.save(r'/encoding/labels/labels.npy', train_names)
        # train_names.clear()
        # train_names.clear()
        # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # locations = fr.face_locations(frame, model='cnn')  # cnn
        # img1 = frame[locations[0][0]:locations[0][2], locations[0][3]:locations[0][1]]
        # encoding = fr.face_encodings(frame, locations, num_jitters=10)[0]
        # new_enc.append(encoding)
        # new_enc = np.array(new_enc)
        # new_name = [label]
        # new_name = np.array(new_name)
        # self.train_faces = np.concatenate((self.train_faces, new_enc), axis=0)
        # self.train_names = np.concatenate((self.train_names, new_name), axis=0)
        # print(np.shape(self.train_faces))
        # self.train_faces = new_enccodings
        # self.train_names = new_names
        # return 'Model Retraining Complete!'

    # def detect_face_locations(image):
        # face_locations = fr.face_locations(image, model='cnn')
        # fr.face_encodings(image)
        # return face_locations
    # def gen_face_enc (image, face_locations):
        # face_enc = fr.face_encodings(image, face_locations)[0]
        # return face_enc
        # face_encoding = np.array(face_enc)
        # np.save(r'encoding/encodings.npy', face_encoding)
    # def retrain(self, image, label):
        # img = fr.load_image_file(image)
        # parent_dir = '/home/ncbc-iftikhar/Facial-Recognition/images/Train'
        # directory = label
        # path = os.path.join(parent_dir, directory)
        # if os.path.isdir(path):
        #     count = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
        #     os.chdir(path)
        #     x = label.strip()
        #     cv2.imwrite(f'{x[0]}{count+1}.jpg', img)
        # else:
        #     os.makedirs(path)
        #     os.chdir(path)
        #     x = label.strip()
        #     cv2.imwrite(f'{x[0]}+1.jpg', img)
        # status = self.update_enc_labels(img, label)
        # return status