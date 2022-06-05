# Arcfelismerő demo
# A kameraképen észlelt arcok azonosítása a tanításkor mentett modell alapján

import cv2

with open('persons.txt') as f:
    labels = f.read().splitlines()
f.close()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('face-trainer.yml')

cap = cv2.VideoCapture(1) # USB Webkamera
while(True):
    ret, img = cap.read()

    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        name = labels[id_]
        text = "{}: {:.2f}%".format(name, conf)
        #if conf >= 50.0:
            #name = labels[id_]
        #else:
            #name = 'ismeretlen'
        if conf <= 100.0:
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(img, text, (x, y-30), font, 0.85, (255,0,0), 2) # BGR
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Recognition Demo', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Erőforrások felszabadítása
cap.release()
cv2.destroyAllWindows()
