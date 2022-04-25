# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:21:24 2022

@author: joaom
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

data = np.load('../calibration/stereo.npz')
cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, Tr, E, F = data['arr_0'], data[
    'arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6'], data['arr_7']

data = np.load('../calibration/rectify.npz')
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = data['arr_0'], data['arr_1'], data[
    'arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6']


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


img_left = glob.glob('../calibration/left/*.png')
img_right = glob.glob('../calibration/right/*.png')
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


# this has to have undistorted image size
map11, map12 = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
map21, map22 = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)

x = []
y = []
z = []
for i in range(0,(len(img_left))):
    frame = cv2.imread(img_left[i])
    frame = cv2.remap(frame, map11, map12, cv2.INTER_AREA)
    frame = cv2.resize(frame, (800,600))
    # frame = frame[200:600,250:700]

    frame_right = cv2.imread(img_right[i])
    frame_right = cv2.remap(frame_right, map21, map22, cv2.INTER_AREA)
    frame_right = cv2.resize(frame_right, (800,600))
    # frame_right = frame_right[200:600,250:700]

    background = mog.apply(frame)
    background_right = mog.apply(frame_right)
        
    mask = np.zeros_like(frame)
    
    contours,_ = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea,reverse= True)

    contours_right,_ = cv2.findContours(background_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_right = sorted(contours_right,key=cv2.contourArea,reverse= True)
    
    cnt = None
    for cnt in contours:
        if cv2.contourArea(cnt) < 4000:
            continue
    center_left = cnt.mean(axis=0)
    cnt_right = None
    for cnt_right in contours_right:
        if cv2.contourArea(cnt_right) < 4000:
            continue
    center_right = cnt_right.mean(axis=0)
    
    if np.linalg.norm(center_left-center_right) < 30:
        print(center_left)
        print(center_right)
        pt3D = cv2.triangulatePoints(P1, P2, center_left[0], center_right[0])
        print(pt3D)
        x.append(pt3D[0][0])
        y.append(pt3D[1][0])
        z.append(pt3D[2][0])

    continue

    (x,y,w,h) = cv2.boundingRect(cnt)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,10),1)
    cv2.putText(frame,f'{BG_method}',(20,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0,2))
    cv2.putText(frame,'Motion Detected',(20,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0,2))
    #cv2.drawContours(background,cnt,-1,255,3)
    cv2.drawContours(mask,cnt,-1,255,3)
    
    mask_n_frame = np.hstack((frame,mask))
    cv2.imshow('image',mask_n_frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

fig = plt.figure()
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
ax.scatter3D(x, y, x)
plt.show()

cv2.destroyAllWindows()