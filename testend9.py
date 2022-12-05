'''
Camera Example
==============

This example demonstrates a simple use of the camera. It shows a window with
a buttoned labelled 'play' to turn the camera on and off. Note that
not finding a camera, perhaps because gstreamer is not installed, will
throw an exception during the kv language processing.

'''

# Uncomment these lines to see all the messages
# from kivy.logger import Logger
# import logging
# Logger.setLevel(logging.TRACE)

#from kivy.app import App
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.floatlayout import FloatLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.toolbar.toolbar import MDTopAppBar
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDTextButton
from kivy.uix.textinput import TextInput
from kivymd.uix.textfield.textfield import MDTextField
from kivymd.uix.dialog import MDDialog
from kivymd.uix.spinner.spinner import MDSpinner
from kivymd.uix.button import MDFlatButton


from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
import time
from kivy.core.audio import SoundLoader

# Import other dependencies
import cv2
import tensorflow as tf
#from layers import L1Dist
import os
import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
from gtts import gTTS

import time


class TestCamera(MDApp):

    def build(self):
        self.message = ""
        self.optionc = False
        self.top_bar = MDTopAppBar(title = "Drowsiness Detection", pos_hint = {'top':1})
        #self.web_cam = Image(pos_hint = {'y':0.5}, size_hint=(1,1/3))
        #self.warn_lab = MDLabel(text = "Warnings", halign = "center", pos_hint = {'top': 0.5},
            #size_hint=(1,None), theme_text_color = "Error")
        #self.textinput = TextInput(text=self.message, pos_hint = {'top': 0.4},
            #size_hint=(1,None), height = '200dp', foreground_color = (245,0,0,1))
        self.start_but = MDRaisedButton(text = 'Start',
            on_press = self.update,
            height = '48dp',
            pos_hint = {'bottom':1, 'x':1/2},
            size_hint=(1/2,None),
            line_color = "white")

        '''self.cap_but = MDRaisedButton(text = 'Capture',
                                    height = '48dp',
                                    pos_hint = {'bottom':1, 'x':0},
                                    size_hint_x = 1/3,
                                    size_hint_y = None,
                                    line_color = "white")'''

        self.ign_but = MDRaisedButton(text = 'Ignore',
            height = '48dp',
            pos_hint = {'bottom':1, 'x':0},
            size_hint=(1/2,None),
            line_color = "white")

        self.spin = MDSpinner(size_hint = (None, None), size = ('30dp', '30dp'),
            pos_hint = {'center_x': .7, 'center_y': .45},
            active = False)

        self.dialog = MDDialog(
                title = "Instruction",
                text = "Messages",
                buttons=[
                    MDFlatButton(
                        text="Cancel",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_press = self.stop
                    ),
                    MDFlatButton(
                        text="Agree",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_press = self.agree
                    )])

        self.dialog2 = MDDialog(
                title = "Camera error",
                text = '''camera access has been denied. 
                Either run 'tccutil reset Camera' command in same terminal to reset application authorization status, 
                either modify 'System Preferences -> Security & Privacy -> Camera' settings for your application.
                camera failed to properly initialize!''',
                buttons=[
                    MDFlatButton(
                        text="Cancel",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        #on_press = self.dialog2.dismiss(force=True)
                    ),
                    MDFlatButton(
                        text="Quit",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_press = self.stop
                    )])
        
        self.dialog.open()

        self.layout = FloatLayout()
        self.layout.add_widget(self.top_bar)
        #layout.add_widget(self.web_cam)
        self.layout.add_widget(self.start_but)
        #layout.add_widget(self.warn_lab)
        #layout.add_widget(self.textinput)
        #layout.add_widget(self.cap_but)
        self.layout.add_widget(self.ign_but)
        #layout.add_widget(self.dia)
        self.sound1 = SoundLoader.load('start.mp3')
        self.sound2 = SoundLoader.load('alert.mp3')
        self.sound3 = SoundLoader.load('yawning.mp3')
        
        return self.layout

    def agree(self, *args):
        self.dialog.dismiss(force=True)

        self.capture = cv2.VideoCapture(0)
        
        if not self.capture.isOpened():
            cap = cv2.VideoCapture(1)
        if not self.capture.isOpened():
            self.dialog2.open()

        if self.capture.isOpened():
            self.web_cam = Image(pos_hint = {'y':0.5}, size_hint=(1,1/3))
            self.textinput = TextInput(text=self.message, pos_hint = {'top': 0.4},
                size_hint=(1,None), height = '200dp', foreground_color = (245,0,0,1))
            self.warn_lab = MDLabel(text = "Warnings", halign = "center", pos_hint = {'top': 0.5},
                size_hint=(1,None), theme_text_color = "Error")
            self.layout.add_widget(self.web_cam)
            self.layout.add_widget(self.textinput)
            self.layout.add_widget(self.warn_lab)
            self.layout.add_widget(self.spin)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            self.event = Clock.schedule_interval(self.normal, 1.0/33.0)

            self.hog_face_detector = dlib.get_frontal_face_detector()

            self.dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

            self.counter_eye = 0
            self.counter_mouth = 0
            self.yawn_thresh = 50
            return self.layout

    def update(self, *args):
        if self.capture.isOpened():
            if self.optionc == True:
                self.start_but.text = "Close"
                self.event.cancel()
                self.event = Clock.schedule_interval(self.start2, 1.0/33.0)
                self.spin.active = True
                #os.system('say "Start detection"')
                self.sound1.play()
            elif self.optionc == False:
                self.event.cancel()
                self.event = Clock.schedule_interval(self.normal, 1.0/33.0)
                self.start_but.text = "Start"
                self.spin.active = False

        
    def normal(self, *args):
        # Read frame from opencv
        ret, frame = self.capture.read()
        #frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
        self.optionc = True

    

    def start2(self, *args):
        self.optionc = False
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
       
        _, frame = self.capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.hog_face_detector(gray)

        for face in faces:

            face_landmarks = self.dlib_facelandmark(gray, face)
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

            if lip_dist > self.yawn_thresh:
                self.counter_mouth = self.counter_mouth + 1
                cv2.putText(frame,"Mouth opend",(20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),3)
                if self.counter_mouth > 10: 
                    cv2.putText(frame, "Yawning detected", (frame.shape[1]//2 - 170, frame.shape[0]//2),
                     cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200), 2)
                    self.message += "Drosiness detected by yawning \n"
                    self.textinput.text = self.message
                    print("Yawn")
                    #os.system('say "Yawning"')
                    self.sound3.play()
                    self.counter_mouth = 0
                '''
                if self.counter_mouth > 10:
                    #playsound('C:\\Users\\사용자\\OneDrive\\바탕 화면\\Testing\\yawning.mp3')
                    #playsound('yawning.mp3')
                    self.counter_mouth = -10'''                
                    
            if EAR<0.26:
                self.counter_eye = self.counter_eye + 1
                cv2.putText(frame,"Eye closed",(20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),3)
                #self.warn_lab.text = "Warnings\nEye closed"
                #print("Drowsy")
                if self.counter_eye > 10:
                    cv2.putText(frame, "ALERT!!", (250, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,7,(0,0,255),3)
                    self.message += "Drosiness detected by eyes \n"
                    self.textinput.text = self.message
                    print("Drowsy")
                    os.system('say "Drosy eyes"')
                    #winsound.Beep(frequency, duration)    
                    self.counter_eye = 0
            print(EAR)

        #print(self.capture.isOpened())
        
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture




if __name__ == '__main__':
    TestCamera().run()