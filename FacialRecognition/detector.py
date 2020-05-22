import cv2,os
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path+r'\trainer\trainer.yml')
cascadePath = path+"\Classifiers\face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font
while True:
    ret, frame =cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x-50,y-50),(x+w+50,y+h+50),(0,255,0),2)
        if(nbr_predicted==0 and conf<=63):
             tag='MM'
        else:
            tag='unknown'
        cv2.putText(frame,tag, (x,y),font, 1.1, (255,0,0), 2, cv2.LINE_AA) #Draw the text
        print(tag)
        print(nbr_predicted)
        print(conf)

    cv2.imshow('frame',frame)
    k=cv2.waitKey(20) & 0xFF
    if k==27 or k==ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()









