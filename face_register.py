import cv2, time

#var kamera
webcam = 0
cam = cv2.VideoCapture(webcam)
#var input nama
id = input('Id :')
name = input('Nama :')
#load file classifier untuk deteksi wajah
faceDetector = cv2.CascadeClassifier('src/haarcascade/haarcascade_frontalface_alt2.xml')
a = 0
#looping
while True :
    a = a+1
    ret, frame = cam.read()
    #menentukan warna gambar yang disimpan
    warna = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceDetector.detectMultiScale(warna, 1.3, 5)
    #jike mendeteksi wajah akan disimpan
    for (x, y, w, h) in faces :
        #lokasi file disimpan
        cv2.imwrite('DataSet/User'+str(id)+'.'+str(name)+'.'+str(a)+'.jpg',warna[y:y+h,x:x+w])
        #memberi kotak pada wajah
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 5)
        rec_face = warna [y : y + w, x : x + w]
    #membuka frame kamera
    cv2.imshow('WEBCAM', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27 :
        break
cam.release()
cv2.destroyAllWindows()