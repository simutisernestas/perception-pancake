# import necessary libraries
import glob
import warnings
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T
from PIL import Image
warnings.filterwarnings('ignore')

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

    for i in range(count):
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
            color = (np.random.randint(0, 255), np.random.randint(
                0, 255), np.random.randint(0, 255))
            thickness = 10
            image = cv2.rectangle(image, start_point,
                                  end_point, color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, pred_class[obj_idx], (10, 100),
                        font, 4, color, 10, cv2.LINE_AA)

        # cv2.imshow('vid1', image)
        # cv2.waitKey(10)
        video.write(image)

    video.release()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
