import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
#import winsound
from imutils import face_utils
from gtts import gTTS
from playsound import playsound
from pydub import AudioSegment
import os
frequency = 2000
duration = 1000


path = '/Users/ykb/Downloads/2022Fall/ds440/frontos/'
start = "Detection test start"
yawning = "Yawning detected"
alert = "Drowsiness detected. Please pull over the car near the street as soon as possible."

audio1 = gTTS(text=start, lang="en", slow=False)
audio2 = gTTS(text=yawning, lang="en", slow=False)
audio3 = gTTS(text=alert, lang="en", slow=False)

audio1.save("start.mp3")
audio2.save("yawning.mp3")
audio3.save("alert.mp3")

#playsound('C:\\Users\\사용자\\OneDrive\\바탕 화면\\Testing\\start.mp3')
#playsound('/Users/ykb/Downloads/2022Fall/ds440/frontos/start.mp3')

def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

def cal_yawn(shape): 
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
  
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
  
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
  
    distance = dist.euclidean(top_mean,low_mean)
    return distance

cap = cv2.VideoCapture(0)
counter_eye = 0
counter_mouth = 0
yawn_thresh = 35
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        shape = face_utils.shape_to_np(face_landmarks)

        lip = shape[48:60]
        cv2.drawContours(frame,[lip],-1,(0, 165, 255), thickness=1)

        leftEye = []
        rightEye = []

        lip_dist = cal_yawn(shape)

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                 next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)

        if lip_dist > yawn_thresh:
            counter_mouth = counter_mouth + 1
            cv2.putText(frame, "Yawning detected", (frame.shape[1]//2 - 170, frame.shape[0]//2),
             cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200), 2)
            print("Yawn")
            if counter_mouth > 10:
                #playsound('C:\\Users\\사용자\\OneDrive\\바탕 화면\\Testing\\yawning.mp3')
                #playsound('yawning.mp3')
                counter_mouth = -10                
                
        if EAR<0.28:
            counter_eye = counter_eye + 1
            cv2.putText(frame,"Drowsiness detected",(20,100),
        		cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            print("Drowsy")
            if counter_eye > 150:
                #playsound('C:\\Users\\사용자\\OneDrive\\바탕 화면\\Testing\\alert.mp3') 
                #playsound('alert.mp3') 
                print("drosy eye")
                counter_eye = 0
        print(EAR)

    

    cv2.imshow("Drowsiness Detection System", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
