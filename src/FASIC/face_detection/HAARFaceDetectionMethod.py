
from FASIC import FaceDetectionMethod 

import cv2 
import numpy as np 

class HAARFaceDetectionMethod(FaceDetectionMethod):

    def detect(self, image):
        res = [] 
        face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            res.append((y,x,h,w))
        return res
