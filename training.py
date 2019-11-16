import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset")

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

current_id = 0
label_ids = {}
y_labels = []
x_trains = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            faces = detector.detectMultiScale(image_array)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_trains.append(roi)
                y_labels.append(id_)


with open("labels.pickle", "wb")as f:
    pickle.dump(label_ids, f)


recognizer.train(x_trains, np.array(y_labels))
print("model trained successfully")
recognizer.save("trainer.yml")