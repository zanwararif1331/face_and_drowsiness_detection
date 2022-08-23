import cv2
import os
import numpy as np
from PIL import Image
import pickle

#menentukan lokasi folder yang digunakan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "src/images2")
#menggunakan classifier untuk mendeteksi wajah
face_cascade = cv2.CascadeClassifier('src/haarcascade/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#menyiapkan variable yang dibutuhkan
current_id = 0
label_ids = {}
y_labels = []
x_train = []
#menentukan file yang akan digunakan pada direktori
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root)
			#setiap folder diberi id dan dijadikan label
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			pil_image = Image.open(path).convert("L")
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			#tiap gambar disimpan ke dalam bentuk array
			image_array = np.array(final_image, "uint8")
			faces = face_cascade.detectMultiScale(image_array)
			#simpan bentuk array ke dalam variable list
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
#simpan data id dan label ke file pickle
with open("src/pickles/face-labels1.pickle", 'wb') as f:
	pickle.dump(label_ids, f)
#simpan data list dalam bentuk file pickle
recognizer.train(x_train, np.array(y_labels))
recognizer.save("src/recognizer/training2.xml")