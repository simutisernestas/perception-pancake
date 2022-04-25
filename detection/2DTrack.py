# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:21:24 2022

@author: joaom
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


###################
###Mask Settings###
###################

min_blue=0 
min_green=36 
min_red=147
max_blue=35 
max_green=127 
max_red=190

####################
##Image Processing##
####################


img_left = glob.glob('StereoNoOcclusion/left/*.png')
img_right = glob.glob('StereoNoOcclusion/right/*.png')
#Rearrange
img_left.sort()
img_right.sort()
#Size of image
img_shape = cv2.imread(img_left[0]).shape
h = img_shape[0]
w = img_shape[1]

####################
#### Background ####
####################

mog = cv2.createBackgroundSubtractorMOG2()  
knn = cv2.createBackgroundSubtractorKNN() 

####################
##### Settings #####
####################


BG_method='MOG' ### MOG or KNN


####################
# Image Processing #
####################

for i in range(0,(len(img_left))):
    frame = cv2.imread(img_left[i])
    frame = cv2.resize(frame, (800,600))
    frame = frame[200:600,250:700]
    if BG_method == 'MOG':
        background = mog.apply(frame)
    elif BG_method == 'KNN':
        background = knn.apply(frame)
        
    mask = np.zeros_like(frame)
    
    contours,_ = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea,reverse= True)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 4000:
                continue
        
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,10),1)
        cv2.putText(frame,f'{BG_method}',(20,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0,2))
        cv2.putText(frame,'Motion Detected',(20,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0,2))
        #cv2.drawContours(background,cnt,-1,255,3)
        cv2.drawContours(mask,cnt,-1,255,3)
        break
    
    mask_n_frame = np.hstack((frame,mask))
    cv2.imshow('image',mask_n_frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cv2.destroyAllWindows()