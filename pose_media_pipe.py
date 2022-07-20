#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:57:18 2022

@author: shaurjyamandal
"""

import cv2 
import mediapipe as mp 
from random import randint

#value = randint(0, 10)
#print(value)
mp_drawing=mp.solutions.drawing_utils 
mp_drawing_styles=mp.solutions.drawing_styles 
mp_pose=mp.solutions.pose

cap=cv2. VideoCapture(0) 
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose: 
    while cap. isOpened():
        success, image = cap.read() 
        image. flags.writeable = False 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        results = pose.process(image)
        image. flags.writeable = True 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        value = randint(0, 100)
        print(value)
        mp_drawing. draw_landmarks (image, results. pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) 
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1)) 
        if cv2.waitKey(5) & 0xFF == 27:
            break 
cap.release()
