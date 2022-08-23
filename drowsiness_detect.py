from tracemalloc import start
import cv2
from tensorflow import keras
import numpy as np
import pickle
from datetime import datetime
from datetime import time
import time
import xlsxwriter

webcam = 1
cam = cv2.VideoCapture(webcam)
date = datetime.now().strftime('%d-%m-%Y')
data = []

def full_face_detection_pipeline():  
    p = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    face_cascade = cv2.CascadeClassifier('src/haarcascade/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('src/haarcascade/haarcascade_eye.xml')
    load_model = keras.models.load_model('src/haarcascade/Driver_Drowsiness_Detection.h5')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('src/recognizer/training2.xml')

    labels_dict={0:'Mengantuk',1:'Tidak Mengantuk'}
    color_dict={0:(0,0,255),1:(0,255,0)}

    labels = {"person_name": 1}
    with open("src/pickles/face-labels1.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    while True :
        size = 4
        ret, frame = cam.read()
        img = cv2.flip(frame,1,1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(img, (img.shape[1] // size, img.shape[0] // size))
        faces = face_cascade.detectMultiScale(mini)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
        cv2.putText(img, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20,30), font, 0.5, (0,255,255))
        for rect in faces:
            (x, y, w, h) = [v * size for v in rect] 
            eyes = eye_cascade.detectMultiScale(img, 1.1, 4)
            for (ex, ey, ew, eh) in eyes:
                eye = img[ey:ey+eh, ex:ex+ew]
                eye = cv2.resize(eye, (32, 32))
                eye = np.array(eye)
                eye = np.expand_dims(eye, axis=0)
                result=load_model.predict(eye)
                print(result)
                if result[0][0] > result[0][1]:
                    percent = round(result[0][0]*100,2)
                else:
                    percent = round(result[0][1]*100,2)
                #var label berdasarkan
                label=np.argmax(result,axis=1)[0]
            
            if result[0][0] < result[0][1]:
                start_time = time.time()
            elif result[0][0] > result[0][1]:
                stop_time = time.time()
                if stop_time - start_time >= 4:
                    #lokasi gambar yang disimpan
                    path = 'src/face_capture/User.'+str(labels[id_])+'.'+str(date)+'.'+str(p+1)+'.jpg'
                    cv2.imwrite(path, img[y:y+h,x:x+w])
                    #menyimpan data dalam array
                    data.append([labels[id_], labels_dict[label], datetime.now().strftime('%d/%m/%Y %H:%M:%S'), path])
                    #jalankan fungsi absen
                    capture()
                    p = p+1
                    start_time = time.time()

            id_, conf = recognizer.predict(gray[y:y+h,x:x+w])
            cv2.putText(img, labels[id_],(x+40,y-20), font, 1, (255,0,0), 2)
            cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
            cv2.rectangle(img,(x,y+h),(x+w,y+h+30),(255,0,0),-1)
            cv2.putText(img, labels_dict[label], (x, y+h+20),font,0.6,(255,255,255),2)

        cv2.imshow('Deteksi Kantuk', img)
        k = cv2.waitKey(30) & 0xff
        #menyimpan data wajah
        #tutup program
        if k == 27:
            break

def capture():
    row = 1
    col = 0
    #menentukan header
    header = ['name', 'status', 'times', 'photos']
    #membuat file excel
    workbook = xlsxwriter.Workbook('src/face_capture/face_captured.xlsx')
    #menentukan bentuk & isi file excel
    worksheet = workbook.add_worksheet("My sheet")
    worksheet.write(0, col, 'name')
    worksheet.write(0, col+1, 'status')
    worksheet.write(0, col+2, 'times')
    worksheet.write(0, col+3, 'photos')
    for name, status, times, photos in (data):
        worksheet.write(row, col, name)
        worksheet.write(row, col + 1, status)
        worksheet.write(row, col + 2, times)
        worksheet.write(row, col + 3, photos)
        row += 1
    workbook.close()

full_face_detection_pipeline()
cam.release()
cv2.destroyAllWindows()