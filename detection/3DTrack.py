# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 00:37:14 2022

@author: joaom
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt



##################
#### Settings ####
##################

img_left = glob.glob('StereoNoOcclusion/left/*.png')
img_right = glob.glob('StereoNoOcclusion/right/*.png')

erode_set = 5
dilate_set = 7
close_set = 55
erode_iter = 2

min_blue=0 
min_green=36 
min_red=147
max_blue=35 
max_green=127 
max_red=190

###################
#### Functions ####
###################


def mask(mask, img):
    
    #Erosion
    cv2.morphologyEx(mask,cv2.MORPH_ERODE, np.ones((erode_set,erode_set),np.uint8),dst=mask,iterations=erode_iter)
    
    #Detection
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_set,close_set),np.uint8),dst=mask)
    
    ret, thresh = cv2.threshold(cv2.GaussianBlur(cv2.bitwise_and(img, mask), (9, 9), 0), 48, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if ret:
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # check only if there is some contour to find
        if len(contours):
            cnt = contours[0]
            max_area = cv2.contourArea(cnt)
            for con in contours:
                tmp_area = cv2.contourArea(con)
                if tmp_area > max_area:
                    cnt = con
                    max_area = tmp_area

            if cnt.sum():
                ret_marker = True

                rect = cv2.minAreaRect(cnt)

                # # get points for rectangle plot
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(roi, [box], 0, 128, 2)
                # get center point
                (marker_y0, marker_x0), (_, _), _ = rect

    return mask, thresh, marker_x0, marker_y0, ret_marker



####################
# Image Processing #
####################

for i in range(0,(len(img_left))):

    frame_left = cv2.imread(img_left[i])
    frame_right = cv2.imread(img_right[i])
    
    #########
    ## HSV ##
    #########
    
    hsv_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2HSV)
    hsv_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2HSV)
    
    ########
    # GRAY #
    ########
    
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)


    #############
    # Blue Mask #
    #############
    
    mask_left = cv2.inRange(hsv_left, (min_blue, min_green, min_red), (max_blue, max_green, max_red))
    mask_right = cv2.inRange(hsv_right, (min_blue, min_green, min_red), (max_blue, max_green, max_red))


    ###########
    ## Remap ##
    ###########
    
    












