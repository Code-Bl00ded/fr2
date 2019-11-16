import cv2
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person name": 1}
with open("labels.pickle", "rb")as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 6)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        id_, conf = recognizer.predict(roi_gray)
        print(conf)
        if conf < 55:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(img, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cv2.imshow('frame', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

