# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:58:42 2022

@author: joaom
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Stereo_conveyor_without_occlusions.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40,)
#tracker = EuclideanDistTracker()
while True:
    ret, frame = cap.read()
    #height, width, _ = frame.shape
    #print(height, width)
    frame = cv2.resize(frame, (800,600))
    roi = frame[200:600,250:700]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (21,21), 0)
    #_, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)


    mask = object_detector.apply(gray)
    _, mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x,y,w,h = cv2.boundingRect(cnt)
        
            cv2.rectangle(gray, (x,y), (x + w, y+h), (0,255,0), 3)
            detections.append([x,y,w,h])
        
    #cv2.imshow("Frame", frame)
    #cv2.imshow("ROI", roi)
    #cv2.imshow("Threshold", threshold)
    cv2.imshow("Gray",gray)
    
    
    key = cv2.waitKey(1)
    if key == 27:
        break
        
    


cv2.destroyAllWindows()