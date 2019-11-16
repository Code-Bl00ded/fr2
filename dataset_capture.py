import cv2
import os
from xlutils.copy import copy
from xlrd import open_workbook
import openpyxl

'''f = open("demofile.txt", "r")
s = f.readline()
print(s)
f.close()
f = open("demofile.txt", "w")
s1 = int(s) + 1
f.write(str(s1))
f.close()


id1 = input('enter your name :')


my_wb = openpyxl.Workbook()
my_sheet = my_wb.active
c1 = my_sheet.cell(row=int(s), column=1)
c1.value = id1
my_wb.save("Book1.xlsx")
print("success")'''

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


face_id = input('enter your name :')
vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
n = "dataset/"+face_id
assure_path_exists(n)


while True:
    _, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("dataset/" + str(face_id) + '/' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('frame', image_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count >= 80:
        print("Successfully Captured")
        break


vid_cam.release()
cv2.destroyAllWindows()