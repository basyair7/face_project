# Image Analysis Project (Face cv)

import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime

# Search data image
path = 'image'
images = []
className = []
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    className.append(os.path.splitext(cls)[0])

# View list in terminal
print(className)

# Image Analysis
def findEncode(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize or Scale an Image
        scale_percent = 0.20
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dimension = (width, height)
        resizedImg = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

        # Encode Face Image
        encode = face_recognition.face_encodings(resizedImg)[0]
        encodeList.append(encode)

    return encodeList

# Create function attendance (Attendance Information)
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # Input name and data time
        # If name not in list data Attendance, then
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')

#markAttendance(className)

# List name if identified in terminal
encodeListKnow = findEncode(images)
print(len(encodeListKnow))

# Webcam vision
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgScan = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgScan = cv2.cvtColor(imgScan, cv2.COLOR_BGR2RGB)

    # face indetify
    faceCurFrame = face_recognition.face_locations(imgScan)
    encodeCurFrame = face_recognition.face_encodings(imgScan, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print(faceDis)
        # Create match index with numpy
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            # identification name
            name = className[matchIndex].upper()
            print(name)
            # Input name in attendance
            markAttendance(name)
            # Create identify box
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            #  Adj font, color box and face loc
            cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img,(x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            # Adj font name identify
            cv2.putText(img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    # Activate the webcam (camera)
    cv2.imshow('Webcam (not filter)',img)
    cv2.waitKey(1)

