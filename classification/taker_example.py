import cv2
from SGDnoPCA import mainno
#from randnoPCA import *
#from PCAlinearSVC import *
from joblib import dump, load

"""principalComponents_ver = pca.transform([img_flat_ver])
print("princ_size", principalComponents_ver.shape)
print(f'Principal components:{principalComponents_ver[0]}')
pred_ver = model.predict(principalComponents_ver)
print("pred", pred_ver)"""


def is_cup(moddo, image):
    mod = moddo
    img = image
    categories = ['cup','else']
    print("Qui")
    print("The predicted image is : "+categories[mod.predict([img])[0]]) #if you dont put [0], it prints [0] instead of 0

    return mod.predict([img])[0]

def bb(moddo, image):
    mod = moddo
    img = image
    categories = ['box','book']
    print("Qui")

    print("The predicted image is : "+categories[mod.predict([img])[0]])

    return mod.predict([img])[0]

def load_classifier():

    classer = mainno()

    mod_cup = classer.model_cup
    mod_bb = classer.model_bb

    return mod_cup, mod_bb

if __name__ == "__main__":

    ##LOAD MODEL##
    mod_cup, mod_bb = load_classifier()

    ##LOAD IMAGE, RESIZE, GRAYSCALE AND FLATTEN IT##
    url = "pic/val/validation_box.jpg"
    image = cv2.imread(url)
    image_rs = cv2.resize(image, (150,150), interpolation=cv2.INTER_AREA)
    image_flat = image_rs.flatten()
    print("image_flat", image_flat.shape)

    #CHECK IF IT IS A CUP OR NOT, (cup, else)
    pred = is_cup(mod_cup, image_flat)
    print("pred", pred)

    #IF IT IS NOT A CUP, CHECK IF IT IS A BOX OR A BOOK, (box, book)
    if pred == 1:
        pred_bb = bb(mod_bb, image_flat)
        print("pred_bb", pred_bb)




    #model_cup = load('model_cup.joblib') #0 for cup, 1 for else
    #model_bb = load('bb_model.joblib') #0 for box, 1 for book
    #pca_cup = load('pca_cup.joblib')
    #pca_bb = load('pca_bb.joblib')
    #ceccho = classifier()
    #print("pca cup",pca_cup)
    #pca_cup, model_cup, pca_bb, model_bb = mainno()
    
