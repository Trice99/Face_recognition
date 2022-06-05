# Arcfelismerő demo
# Tanítás előre osztályozott képekkel
# Megkeresi az arcokat a képeken és egy modellben tárolja azokat

import cv2
import numpy as np
import os
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

x_train = [] # tanító minták (képek)
y_train = [] # címkék (nevek)

Face_ID = -1
prev_person_name = ''

Face_Images = os.path.join(os.getcwd(), 'Face_Images')
print (Face_Images)

def deleteContent(pfile):
    pfile.seek(0)
    pfile.truncate()
            
with open('persons.txt', 'w') as f:
    deleteContent(f)
f.close()

for root, dirs, files in os.walk(Face_Images):
    for file in files:
        if file.endswith('jpeg') or file.endswith('jpg') or file.endswith('png') or file.endswith('NEF') or file.endswith('JPG'):
            path = os.path.join(root, file)
            person_name = os.path.basename(root)
            print(path, '\n', person_name)
            
            if prev_person_name != person_name:
                Face_ID = Face_ID + 1
                prev_person_name = person_name
                with open('persons.txt', 'a') as f:
                    f.write(person_name + '\n')
                f.close()
            
            print('Face_ID: ', Face_ID)
            
            Grey_Image = Image.open(path).convert('L')
            Crop_Image = Grey_Image.resize((300,300), Image.ANTIALIAS)
            Final_Image = np.array(Crop_Image).astype('uint8')
            # Arc keresése a képen
            faces = face_cascade.detectMultiScale(Final_Image) # default: scaleFactor=1.1, minNeighbors=3
            print(faces)
            
            for (x,y,w,h) in faces:
                roi = Final_Image[y:y+h, x:x+w]
                x_train.append(roi)
                y_train.append(Face_ID)

recognizer.train(x_train, np.array(y_train))
recognizer.save('face-trainer.yml')