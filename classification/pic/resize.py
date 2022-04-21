import matplotlib as plt
import mpl_toolkits
import os
import cv2

def res(ratio):

    list = []
    
    for i in range(1, 201):
        print("foto pic ", i)
    
        im = cv2.imread("C:/Users/albor/OneDrive - Danmarks Tekniske Universitet/Dokumenter/PFAS/project/friendly-pancake/classification/pic/books/pics/pic{}".format(i),cv2.IMREAD_UNCHANGED)   
        #im = cv2.imread("C:/Users/lorec/friendly-pancake/classification/pic/books/pic{}.jpg".format(i),cv2.IMREAD_UNCHANGED)

        scale = ratio # percent of original size
        width = int(im.shape[1] * scale)
        height = int(im.shape[0] * scale)
        dim = (width, height)

        im_resized = cv2.resize(im,dim,interpolation = cv2.INTER_AREA)
        
        filename = "C:/Users/albor/OneDrive - Danmarks Tekniske Universitet/Dokumenter/PFAS/project/friendly-pancake/classification/pic/books/pics_resized/pic_res{}.jpg".format(i)        
        #filename = "C:/Users/lorec/friendly-pancake/classification/pic/books/pic_res{}.jpg".format(i)
        cv2.imwrite(filename, im_resized)

        list.append(im_resized)

    return list


def main():
    ratio = 0.20

    pics = res(ratio)

    print("lista size {}".format(len(pics)))


if __name__=="__main__":
    main()