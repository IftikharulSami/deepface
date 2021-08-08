import faiss
from deepface import DeepFace
import numpy as np
import os

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
        self.index.add(self.train_emb)
        D, I = self.index.search(tst_emb, 3)
        idx = I[0][0]
        distance = D[0][0]
        label = self.train_names[idx]
        return label, distance
