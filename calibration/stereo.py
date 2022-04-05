import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

img = cv2.imread('data/left-0000.png')
h, w = img.shape[:2]

data = np.load('left.npz')
left_camera_matrix, left_dist_coefs = data['arr_0'], data['arr_1']
left_newcameramtx, left_roi = cv2.getOptimalNewCameraMatrix(
    left_camera_matrix, left_dist_coefs, (w, h), 1, (w, h))

data = np.load('right.npz')
right_camera_matrix, right_dist_coefs = data['arr_0'], data['arr_1']
right_newcameramtx, right_roi = cv2.getOptimalNewCameraMatrix(
    right_camera_matrix, right_dist_coefs, (w, h), 1, (w, h))

pattern_size = (9, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= 1.0

obj_points = []
left_img_points = []
right_img_points = []

right_img_files = glob.glob('data/right*')
right_img_files.sort()
left_img_files = glob.glob('data/left*')
left_img_files.sort()

# print(right_img_files[:10],left_img_files[:10])
# exit()

for lmn, rmn in zip(right_img_files, left_img_files):
    limg = cv2.imread(lmn, cv2.IMREAD_GRAYSCALE)
    rimg = cv2.imread(rmn, cv2.IMREAD_GRAYSCALE)

    assert w == limg.shape[1] and h == limg.shape[0], ("size: %d x %d ... " % (
        img.shape[1], img.shape[0]))
    assert w == rimg.shape[1] and h == rimg.shape[0], ("size: %d x %d ... " % (
        img.shape[1], img.shape[0]))

    found, lcorners = cv2.findChessboardCorners(limg, pattern_size)
    if not found:
        print("not found!")
        continue
    found, rcorners = cv2.findChessboardCorners(rimg, pattern_size)
    if not found:
        print("not found!")
        continue

    # if found:
    #     lterm = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    #     cv2.cornerSubPix(limg, rcorners, (5, 5), (-1, -1), lterm)
    #     rterm = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    #     cv2.cornerSubPix(rimg, lcorners, (5, 5), (-1, -1), rterm)

    left_img_points.append(lcorners.reshape(-1, 2))
    right_img_points.append(rcorners.reshape(-1, 2))
    obj_points.append(pattern_points)

flags = 0
# flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
# flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
rms, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, left_img_points, right_img_points, left_newcameramtx, left_dist_coefs, right_newcameramtx, right_dist_coefs, (w, h), criteria=criteria, flags=flags)

print("RMS:", rms)
if rms < 1.0:
    np.savez('stereo.npz', cameraMatrix1, distCoeffs1,
             cameraMatrix2, distCoeffs2, R, T, E, F)
else:
    exit()

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)
P2
np.savez('rectify.npz', R1, R2, P1, P2, Q, validPixROI1, validPixROI2)

# this has to have undistorted image size
map11, map12 = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
map21, map22 = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)

img1 = cv2.imread('left/1585434283_489372015_Left.png')
dst1 = cv2.remap(img1, map11, map12, cv2.INTER_AREA)
dst1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('right/1585434283_489372015_Right.png')
dst2 = cv2.remap(img2, map21, map22, cv2.INTER_AREA)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)

plt.figure(1, figsize=(10, 10))
plt.imshow(dst1, cmap='gray')
plt.figure(2, figsize=(10, 10))
plt.imshow(dst2, cmap='gray')
plt.show()

# triangulation
# https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
