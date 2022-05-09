from calendar import c
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import cv2
#from tester import *
#from randnoPCA import *
#from PCAlinearSVC import *
from joblib import dump, load
from sklearn.decomposition import PCA

"""principalComponents_ver = pca.transform([img_flat_ver])
print("princ_size", principalComponents_ver.shape)
print(f'Principal components:{principalComponents_ver[0]}')
pred_ver = model.predict(principalComponents_ver)
print("pred", pred_ver)"""


def is_cup(moddo, img, pca_cup):
    model = moddo
    categories = ['cup','else']
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resize=resize(img_gray,(150,150))
    #blue,green,red = cv2.split(img_resize) 
    #img_flat=img_resize.flatten()
    print("Qui")

    pca_cup = pca_cup
    principalComponents_cup = pca_cup.fit_transform(img_resize)
    
    print("princ_size", principalComponents_cup.shape)
    print(f'Principal components:{principalComponents_cup[0]}')
    print(pca_cup.explained_variance_ratio_)
    print(pca_cup.explained_variance_)
    probability=model.predict_proba(principalComponents_cup)
    for ind,val in enumerate(categories):
        print(f'{val} = {probability[0][ind]*100}%')
    print("The predicted image is : "+categories[model.predict(principalComponents_cup)[0]])

    return model.predict(principalComponents_cup)[0]

def bb(pca_bb,moddo, img):
    model = moddo
    categories = ['box','book']
    #plt.imshow(img)
    #plt.show()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resize=resize(img_gray,(150,150))
    #img_flat=img_resize.flatten()
    print("Qui")

    pca_bb = PCA(20)

    principalComponents_bb = pca_bb.transform([img_resize])

    print("princ_size", principalComponents_bb.shape)
    print(f'Principal components:{principalComponents_bb[0]}')
    print(pca_bb.explained_variance_ratio_)
    print(pca_bb.explained_variance_)
    probability=model.predict_proba(principalComponents_bb)
    for ind,val in enumerate(categories):
        print(f'{val} = {probability[0][ind]*100}%')
    print("The predicted image is : "+categories[model.predict(principalComponents_bb)[0]])

    return model.predict(principalComponents_bb)[0]

if __name__ == "__main__":
    #model, categorie = classerRAND() #comment/uncomment for randnopca
    #model, categorie = classerSVC() #comment/uncomment for pcalinearsvc

    model_cup = load('model_cup.joblib') #0 for cup, 1 for else
    model_bb = load('bb_model.joblib') #0 for box, 1 for book
    pca_cup = load('pca_cup.joblib')
    pca_bb = load('pca_bb.joblib')
    #ceccho = classifier()
    #print("pca cup",pca_cup)
    #pca_cup, model_cup, pca_bb, model_bb = mainno()
    
    #url=input('Enter URL of Image :')
    url = "pic/val/validation_book.jpg"
    
    image=imread(url)
    plt.imshow(image)
    plt.show()
    pred = is_cup(model_cup, image, pca_cup)
    print("pred", pred)

    if pred == 1:
        pred_bb = bb(pca_bb,model_bb, image)
        print("pred_bb", pred_bb)
