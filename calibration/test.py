import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('../maskrcnn/video.avi')
ret, frame = cap.read()
h,w = frame.shape[:2]

data = np.load('../calibration/stereo.npz')
cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, Tr, E, F = data['arr_0'], data[
    'arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6'], data['arr_7']

data = np.load('../calibration/rectify.npz')
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = data['arr_0'], data['arr_1'], data[
    'arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6']

# this has to have undistorted image size
map11, map12 = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)

img1 = frame
dst1 = cv2.remap(img1, map11, map12, cv2.INTER_AREA)
dst1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)

plt.figure(1, figsize=(10, 10))
plt.imshow(dst1, cmap='gray')
plt.show()
