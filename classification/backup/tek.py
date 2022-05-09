import cv2
from lory import *

#############
#CALL load_classifier() ONCE IN THE DETECTION CODE, 
#THEN model_cup AND model_bb CAN BE USED ANYTIME TO PREDICT NEW IMAGES
#############


def is_cup(moddo, image, pca_cup):
    mod = moddo
    img = image
    pca = pca_cup
    categories = ['cup','else']
    print("Qui")

    X_scaled  =  StandardScaler().fit_transform([image])
    
    principalComponents  =  pca.transform([image])

    print("The predicted image is : "+categories[mod.predict([principalComponents])[0]]) #if you dont put [0], it prints [0] instead of 0

    return mod.predict([img])[0]

def bb(moddo, image, pca_bb):
    mod = moddo
    img = image
    pca = pca_bb
    categories = ['box','book']
    print("Qui")

    print("The predicted image is : "+categories[mod.predict([img])[0]])

    return mod.predict([img])[0]

def load_classifier():

    classer = mainno()

    mod_cup = classer.model_cup
    mod_bb = classer.model_bb
    pca_cup = classer.pca_cup
    pca_bb = classer.pca_bb

    return mod_cup, mod_bb, pca_cup, pca_bb

if __name__ == "__main__":

    ##LOAD MODEL##
    mod_cup, mod_bb, pca_cup, pca_bb = load_classifier()

    url = "pic/val/validation_cup.jpg"
    image = cv2.imread(url)
    image_rs = cv2.resize(image, (150,150), interpolation=cv2.INTER_AREA)
    #img_gray = cv2.cvtColor(image_rs, cv2.COLOR_BGR2GRAY)
    image_flat = image_rs.flatten()
    print("image_flat", image_flat.shape)

    #CHECK IF IT IS A CUP OR NOT, (cup, else)
    pred = is_cup(mod_cup, image_flat, pca_cup)
    print("pred", pred)

    #IF IT IS NOT A CUP, CHECK IF IT IS A BOX OR A BOOK, (box, book)
    if pred == 1:
        pred_bb = bb(mod_bb, image_flat, pca_bb)
        print("pred_bb", pred_bb)

