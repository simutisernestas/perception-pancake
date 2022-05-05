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

min_blue = 0
min_green = 36
min_red = 147
max_blue = 35
max_green = 127
max_red = 190

####################
##Image Processing##
####################

img_left = glob.glob('../shortvideo/*Left.png')
img_right = glob.glob('../shortvideo/*Right.png')
# Rearrange
img_left.sort()
img_right.sort()
# Size of image
img_shape = cv2.imread(img_left[0]).shape
h = img_shape[0]
w = img_shape[1]

####################
#### Background ####
####################

knn = cv2.createBackgroundSubtractorKNN(
    history=500, dist2Threshold=500, detectShadows=False)

####################
##### Settings #####
####################

# this has to have undistorted image size
map11, map12 = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
map21, map22 = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)

####################
# Kalman filtering #
####################


def update(x, P, Z, H, R):
    y = Z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.pinv(S)
    Xprime = x + K @ y
    KH = K @ H
    Pprime = (np.eye(KH.shape[0]) - KH) @ P
    return (Xprime, Pprime)


def predict(x, P, F, u):
    Xprime = F @ x + u
    Pprime = F @ P @ F.T
    return (Xprime, Pprime)


# Load the video
cap = cv2.VideoCapture('../maskrcnn/video.avi')
if not cap.isOpened():
    print("Cannot open video")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/fps

### Initialize Kalman filter ###
# The initial state (6x1).
# x y z x_dt y_dt z_dt x_dt2 y_dt2 z_dt2
state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]).T

# The initial uncertainty (6x6).
P = np.eye(9, 9) * 1000

# The external motion (6x1).
u = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]).T

# Jacobian
F = np.eye(9)
F[0][3] = dt
F[1][4] = dt
F[2][5] = dt
F[0][6] = (1/2) * dt**2
F[1][7] = (1/2) * dt**2
F[2][8] = (1/2) * dt**2

# The observation matrix (2x6).
H = np.zeros((3, 9))
H[0][0] = 1
H[1][1] = 1
H[2][2] = 1

# The measurement uncertainty.
R = 10

####################
# Classification #
####################
# TODO: IMPLEMENT!

def classify(img):
    return np.random.choice([0, 1, 2])

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 580)
fontScale = .75
fontColor = (200, 0, 188)
thickness = 3
lineType = 2

label_to_class = {
    0: "cup",
    1: "book",
    2: "box",
}

####################
# Image Processing #
####################
# x = []
# y = []
# z = []
init = False
state_buffer = []
for i in range(0, (len(img_left))):
    if i > 250:
        break
    frame = cv2.imread(img_left[i])
    frame = cv2.remap(frame, map11, map12, cv2.INTER_AREA)
    frame = cv2.resize(frame, (800, 600))

    frame_right = cv2.imread(img_right[i])
    frame_right = cv2.remap(frame_right, map21, map22, cv2.INTER_AREA)
    frame_right = cv2.resize(frame_right, (800, 600))

    background = knn.apply(frame)
    background_right = knn.apply(frame_right)

    mask = np.zeros_like(frame)
    mask_right = np.zeros_like(frame_right)

    contours, _ = cv2.findContours(
        background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contours_right, _ = cv2.findContours(
        background_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_right = sorted(contours_right, key=cv2.contourArea, reverse=True)

    cnt = contours[0]
    center_left = cnt.mean(axis=0)
    cnt_right = contours_right[0]
    center_right = cnt_right.mean(axis=0)
    
    updated = False
    if cv2.contourArea(cnt) > 1500 and cv2.contourArea(cnt) < 20000:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # print((x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 10), 10)
        (x, y, w, h) = cv2.boundingRect(cnt_right)
        # print((x, y, w, h))
        cv2.rectangle(frame_right, (x, y), (x+w, y+h), (0, 255, 10), 10)
        # draw center point
        # cv2.circle(frame, (int(center_left[0][0]), int(
        #     center_left[0][1])), 30, (0, 0, 255), -1)
        # cv2.circle(frame_right, (int(center_right[0][0]), int(
        #     center_right[0][1])), 30, (0, 0, 255), -1)
        # we kinda expect two centers would close to each other
        # in right and left images : ))))
        if np.linalg.norm(center_left-center_right) < 100.0:
            pt3D = cv2.triangulatePoints(
                P1, P2, center_left[0], center_right[0])
            Z = pt3D[:3]
            state, P = update(state, P, Z, H, R)
            init = True
            updated = True
        # else:
        #     print(f"dist: {np.linalg.norm(center_left-center_right)}")

    # increase uncertainty over time
    if not updated:
        P += np.eye(9,9) * 100
    print(f"unc: {np.linalg.norm(P[:3,:3])}")

    if init:
        state, P = predict(state, P, F, u)
        state_buffer.append(state)

    fontColor = (200, 0, 188)
    bottomLeftCornerOfText = (10, 580)
    cv2.putText(frame, f"x: {state[0]}, y: {state[1]}, z: {state[2]}",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    
    label = classify(frame)
    fontColor = (0, 0, 188)
    bottomLeftCornerOfText = (10, 550)
    cv2.putText(frame, f"Label: {label_to_class[label]}",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    mask_n_frame = np.hstack((frame, frame_right))
    # mask_n_frame = frame
    # mask_n_frame = np.hstack((frame,mask))
    cv2.imshow('image', mask_n_frame)
    cv2.waitKey(10)

# (N, 9)
state_buffer = np.array(state_buffer)

fig = plt.figure()
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
ax.scatter3D(state_buffer[:, 0], state_buffer[:, 1], state_buffer[:, 2])
plt.show()

cv2.destroyAllWindows()
