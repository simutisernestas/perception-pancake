# import necessary libraries
import glob
import warnings
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T
from PIL import Image
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
CONFIDENCE = .8

# mtx1, dist1 = np.load("calibration/left.npz").items()
# mtx1, dist1 = mtx1[1], dist1[1]
# mtx2, dist2 = np.load("calibration/right.npz").items()
# mtx2, dist2 = mtx2[1], dist2[1]

data = np.load('../calibration/stereo.npz')
cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, Tr, E, F = data['arr_0'], data[
    'arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6'], data['arr_7']

data = np.load('../calibration/rectify.npz')
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = data['arr_0'], data['arr_1'], data[
    'arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6']

# #RT matrix for C1 is identity.
# RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
# P1 = mtx1 @ RT1 #projection matrix for C1
# #RT matrix for C2 is the R and T obtained from stereo calibration.
# RT2 = np.concatenate([R, Tr], axis = -1)
# P2 = mtx2 @ RT2 #projection matrix for C2

def process_pred(pred, confidence):
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    try:
        pred_t = [pred_score.index(x)
                  for x in pred_score if x > confidence][-1]
    except:
        return None, None, None
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_CLASS_NAMES[i]
                  for i in list(pred[0]['labels'].detach().cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def main():
    # images from video without occlusions!
    rims = glob.glob("stereo/*Right*")
    rims.sort()
    lims = glob.glob("stereo/*Left*")
    lims.sort()
    count = len(rims)

    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to('cuda')
    # set to evaluation mode
    model.eval()

    width, height = 1280, 720

    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('video.avi', fourcc, 30.0, (width, height))
    transform = T.Compose([T.ToTensor()])

    for i in range(300, count):
        imfile = rims[i]

        imgorg = Image.open(imfile)
        img = transform(imgorg).cuda()
        pred = model([img])
        _, pred_boxes, pred_class = process_pred(pred, CONFIDENCE)

        image = cv2.cvtColor(np.array(imgorg), cv2.COLOR_RGB2BGR)
        if pred_boxes is not None and ('book' in pred_class or 'cup' in pred_class):
            obj_in_frame = []
            if 'book' in pred_class:
                obj_in_frame.append('book')
            if 'cup' in pred_class:
                obj_in_frame.append('cup')

            # just taking the first one, migth wanna do smth else here
            obj_idx = pred_class.index(obj_in_frame[0])

            start_point = (int(pred_boxes[obj_idx][0][0]), int(
                pred_boxes[obj_idx][0][1]))
            end_point = (int(pred_boxes[obj_idx][1][0]), int(
                pred_boxes[obj_idx][1][1]))

            left_correspondanceorg = Image.open(lims[i])
            left_correspondance = transform(left_correspondanceorg).cuda()
            left_pred = model([left_correspondance])
            _, left_pred_boxes, left_pred_class = process_pred(
                pred, CONFIDENCE)

            if left_pred_class[obj_idx] == pred_class[obj_idx]:
                left_start_point = (int(left_pred_boxes[obj_idx][0][0]), int(
                    left_pred_boxes[obj_idx][0][1]))
                left_end_point = (int(left_pred_boxes[obj_idx][1][0]), int(
                    left_pred_boxes[obj_idx][1][1]))
                left_image = cv2.cvtColor(np.array(left_correspondanceorg), cv2.COLOR_RGB2BGR)

                # TODO: This must be done before the prediction!
                # map11, map12 = cv2.initUndistortRectifyMap(
                #     cameraMatrix1, distCoeffs1, R1, P1, (width, height), cv2.CV_32FC1)
                # map21, map22 = cv2.initUndistortRectifyMap(
                #     cameraMatrix2, distCoeffs2, R2, P2, (width, height), cv2.CV_32FC1)
                # dst1 = cv2.remap(left_image, map11, map12, cv2.INTER_AREA)
                # dst1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
                # dst2 = cv2.remap(image, map21, map22, cv2.INTER_AREA)
                # dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)

                # TODO: make it smaller or just use the segMASK!
                # left_start_point = (left_start_point[0]+25,left_start_point[1]+25)
                # left_end_point = (left_end_point[0]-25,left_end_point[1]-25)

                crop_img = left_image[left_start_point[1]:left_start_point[1]+(left_end_point[1]-left_start_point[1]),
                                      left_start_point[0]:left_start_point[0]+(left_end_point[0]-left_start_point[0])]
                right_crop = image[start_point[1]:start_point[1]+(end_point[1]-start_point[1]),
                                   start_point[0]:start_point[0]+(end_point[0]-start_point[0])]

                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY), None)
                kp2, des2 = sift.detectAndCompute(cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY), None)
                kp_img1 = cv2.drawKeypoints(crop_img, kp1, crop_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                kp_img2 = cv2.drawKeypoints(right_crop, kp2, right_crop, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # TODO: visualizing both pred boxes would help!
                cv2.imshow("keypoints1", kp_img1)
                cv2.imshow("keypoints2", kp_img2)

                bf = cv2.BFMatcher()
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                nb_matches = 10
                pts1 = []
                pts2 = []
                for m in matches[:nb_matches]:
                    pts1.append(kp1[m.queryIdx].pt)
                    pts2.append(kp2[m.trainIdx].pt)
                pts1 = np.array(pts1, dtype=np.int32)
                pts2 = np.array(pts2, dtype=np.int32)

                # TODO: these don't lie on the objects
                pts1 = pts1 + np.array([left_start_point[0],left_start_point[1]])
                pts2 = pts2 + np.array([start_point[0],start_point[1]])

                print("pts1", pts1)
                print("pts2", pts2)

                for p in pts1:
                    left_image = cv2.circle(left_image, (p[0],p[1]), radius=5, color=(255,0,0), thickness=3)
                for p in pts2:
                    image = cv2.circle(image, (p[0],p[1]), radius=5, color=(255,0,0), thickness=3)
                cv2.imshow("points_right",image)
                cv2.imshow("points_left",left_image)
                cv2.waitKey(0)
                exit()

                # if you average them out kinda gives a point on the object :) 

                x = []
                y = []
                z = []
                for p1,p2 in zip(pts1,pts2):
                    pt3D =  cv2.triangulatePoints(P1, P2, p2, p1)
                    pt3D = pt3D / pt3D[-1]
                    x.append(pt3D[0][0])
                    y.append(pt3D[1][0])
                    z.append(pt3D[2][0])

                fig = plt.figure()
                ax = plt.axes(projection='3d')
                # Data for a three-dimensional line
                ax.scatter3D(x, y, x)
                plt.show()

                cv2.imshow("left_crop", crop_img)
                cv2.imshow("left", left_image)
                cv2.imshow("right", image)
                cv2.imshow("right_crop", right_crop)
                cv2.waitKey(0)
            else:
                print("doesn't match!")
                exit()

            color = (np.random.randint(0, 255), np.random.randint(
                0, 255), np.random.randint(0, 255))
            thickness = 10
            image = cv2.rectangle(image, start_point,
                                  end_point, color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, pred_class[obj_idx], (10, 100),
                        font, 4,     color, 10, cv2.LINE_AA)

        # cv2.imshow('vid1', image)
        # cv2.waitKey(10)
        video.write(image)

    video.release()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
