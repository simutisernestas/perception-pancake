import os
import cv2
from cv2 import INTER_AREA

def rename(folder):
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"pic{str(count+1)}.jpg"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)

def resgray(folder):
    ratio = 0.20
    for count, filename in enumerate(os.listdir(folder)):
        print("foto pic ", count+1)
        dire = "pic/train/book/pic{}.jpg".format(count+1)
        img_ver = cv2.imread(dire)
        img_res = cv2.resize(img_ver, (150,150), interpolation=INTER_AREA)
        img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        scale = ratio # percent of original size
        width = 150 #int(im.shape[1] * scale)
        height = 150 #int(im.shape[0] * scale)
        dim = (width, height)

        #im_resized = cv2.resize(im_gray, (150,150), interpolation=INTER_AREA)
        
        filename = "pic/train/book/pic_res{}.jpg".format(count+1)
        cv2.imwrite(filename, img_gray)

if __name__ == "__main__":
    folder = "pic/train/book"
    #rename(folder)
    resgray(folder)

