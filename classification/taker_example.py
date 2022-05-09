import cv2
from SGDnoPCA import mainno
from mixedHOG import mainhog
#############
#CALL load_classifier() ONCE IN THE DETECTION CODE, 
#THEN model_cup AND model_bb CAN BE USED ANYTIME TO PREDICT NEW IMAGES
#############


def is_cup(moddo, image):
    mod = moddo
    img = image
    categories = ['cup','else']
    print("Qui")
    print("The predicted image is : "+categories[mod.predict(img)[0]]) #if you dont put [0], it prints [0] instead of 0
    #mod.predict([img])[0] for SGD

    return mod.predict(img)[0]#[] for SGD

def bb(moddo, image):
    mod = moddo
    img = image
    categories = ['box','book']
    print("Qui")

    print("The predicted image is : "+categories[mod.predict(img)[0]])

    return mod.predict(img)[0]

def load_classifier_hog():

    classer_hog = mainhog()

    mod_cup = classer_hog.sgd_clf_cup
    mod_bb = classer_hog.sgd_clf_bb
    grayify = classer_hog.grayify
    hogify = classer_hog.hogify
    scalify = classer_hog.scalify

    return mod_cup, mod_bb, grayify, hogify, scalify

def load_classifier():
    classer = mainno()

    mod_cup = classer.sgd_clf_cup
    mod_bb = classer.sgd_clf_bb
    
    return mod_cup, mod_bb

if __name__ == "__main__":

    """#####SGD#####
    ##LOAD MODEL##
    mod_cup, mod_bb = load_classifier()

    ##LOAD IMAGE, RESIZE, GRAYSCALE AND FLATTEN IT##
    url = "pic/val/validation_cup.jpg"
    image = cv2.imread(url)
    image_rs = cv2.resize(image, (150,150), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(image_rs, cv2.COLOR_BGR2GRAY)
    image_flat = img_gray.flatten()
    print("image_flat", image_flat.shape)

    #CHECK IF IT IS A CUP OR NOT, (cup, else)
    pred = is_cup(mod_cup, image_flat)
    print("pred", pred)

    #IF IT IS NOT A CUP, CHECK IF IT IS A BOX OR A BOOK, (box, book)
    if pred == 1:
        pred_bb = bb(mod_bb, image_flat)
        print("pred_bb", pred_bb)"""


    ########HOG#########
    mod_cup, mod_bb, grayify, hogify, scalify = load_classifier_hog()

    url = "pic/val/validation_cup.jpg"
    image = cv2.imread(url)
    im_rs = cv2.resize(image, (150,150), interpolation=cv2.INTER_AREA)
    
    image_gray = grayify.transform(im_rs)
    image_hog = hogify.transform([image_gray])
    image_prepared = scalify.transform(image_hog)

    pred = is_cup(mod_cup, image_prepared)
    print("pred", pred)

    #IF IT IS NOT A CUP, CHECK IF IT IS A BOX OR A BOOK, (box, book)
    if pred == 1:
        pred_bb = bb(mod_bb, image_prepared)
        print("pred_bb", pred_bb)

