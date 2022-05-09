import os
from cv2 import INTER_AREA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from skimage.feature import hog
import skimage.transform
import timeit
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
import cv2
from joblib import dump

##########
#RGB2GrayTransformer and RGB2GrayTransformer prepare the data
#classifier creates the classifier, the commented functions (see the init function) are for testing.
#LinearSVC seems to perform better to distinguish cups and boxes, while SGDClassifier seems to be better for boxes against books
##########

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])

class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

class classifier():
    def __init__(self):
        super().__init__()
        
        self.cup_model()
        self.bb_model()
        #self.verify()
        self.book_verify()

    def cup_model(self):
        categories = ['cup','else']
        flat_data_arr = [] #input array
        target_arr = [] #output array
        datadir = 'pic/train' 
        #path which contains all the categories of images
        for i in categories:
            print(f'loading... category : {i}')
            path = os.path.join(datadir,i).replace("\\","/")
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img))
                img_res = cv2.resize(img_array, (150,150), interpolation=cv2.INTER_AREA)
                flat_data_arr.append(img_res)
                target_arr.append(categories.index(i))
            print(f'loaded category:{i} successfully')
        X = np.array(flat_data_arr)
        y = np.array(target_arr)
        
        start  =  timeit.default_timer()

        X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        shuffle=True,
        random_state=42,
        )

        # create an instance of each transformer
        self.grayify = RGB2GrayTransformer()
        self.hogify = HogTransformer(
            pixels_per_cell=(14, 14), 
            cells_per_block=(2,2), 
            orientations=9, 
            block_norm='L2-Hys'
        )
        self.scalify = StandardScaler()
        
        # call fit_transform on each transform converting X_train step by step
        X_train_gray = self.grayify.fit_transform(X_train)
        X_train_hog = self.hogify.fit_transform(X_train_gray)
        X_train_prepared = self.scalify.fit_transform(X_train_hog)
         
        print(X_train_prepared.shape)

        weights = {0 : 0.4, 1 : 0.6}
        #self.sgd_clf_cup = LinearSVC(random_state=42,max_iter=2000, tol=1e-8, class_weight=weights)
        self.sgd_clf_cup = SGDClassifier(random_state=42,max_iter=2000, tol=1e-8)
        self.sgd_clf_cup.fit(X_train_prepared, y_train)

        X_test_gray = self.grayify.transform(X_test)
        X_test_hog = self.hogify.transform(X_test_gray)
        X_test_prepared = self.scalify.transform(X_test_hog)

        y_pred = self.sgd_clf_cup.predict(X_test_prepared)
        print(np.array(y_pred == y_test)[:25])
        print('')
        print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

        stop  =  timeit.default_timer() 
        print("Time needed: ", stop-start)   

    def bb_model(self):
        categories = ['book','box']
        flat_data_arr = [] #input array
        target_arr = [] #output array
        datadir = 'pic/train' 
        #path which contains all the categories of images
        for i in categories:
            print(f'loading... category : {i}')
            path = os.path.join(datadir,i).replace("\\","/")
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img))
                img_res = cv2.resize(img_array, (150,150), interpolation=cv2.INTER_AREA)
                flat_data_arr.append(img_res)
                target_arr.append(categories.index(i))
            print(f'loaded category:{i} successfully')
        X = np.array(flat_data_arr)
        y = np.array(target_arr)
        
        start  =  timeit.default_timer()

        X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        shuffle=True,
        random_state=42,
        )
        
        # call fit_transform on each transform converting X_train step by step
        X_train_gray = self.grayify.transform(X_train)
        X_train_hog = self.hogify.transform(X_train_gray)
        X_train_prepared = self.scalify.transform(X_train_hog)
        
        print(X_train_prepared.shape)

        self.sgd_clf_bb = SGDClassifier(random_state=42, max_iter=1500, tol=1e-5)
        self.sgd_clf_bb.fit(X_train_prepared, y_train)

        X_test_gray = self.grayify.transform(X_test)
        X_test_hog = self.hogify.transform(X_test_gray)
        X_test_prepared = self.scalify.transform(X_test_hog)

        y_pred = self.sgd_clf_bb.predict(X_test_prepared)
        print(np.array(y_pred == y_test)[:25])
        print('')
        print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

        stop  =  timeit.default_timer() 
        print("Time needed: ", stop-start)

    def verify(self):
        imlist = []
        ##LOAD IMAGE, RESIZE, GRAYSCALE AND FLATTEN IT##
        url = "pic/val/validation_book.jpg"
        url2 = "pic/val/validation_cup.jpg"
        url3 = "pic/val/validation_box.jpg"


        image = cv2.imread(url)
        image_rs = cv2.resize(image, (150,150), interpolation=cv2.INTER_AREA)
        #image_gray = cv2.cvtColor(image_rs, cv2.COLOR_BGR2GRAY)

        image_gray = self.grayify.transform(image_rs)
        image_hog = self.hogify.transform([image_gray])
        image_prepared = self.scalify.transform(image_hog)

        pred_ver1 = self.sgd_clf_cup.predict(image_prepared)
        print("pred", pred_ver1)
    
        if pred_ver1 == 1:
            pred_ver1_fin = self.sgd_clf_bb.predict(image_prepared)
            print("prob pred", self.sgd_clf_bb.predict_proba(image_prepared))
            print("Final Predction", pred_ver1_fin)


        image2 = cv2.imread(url2)
        image_rs2 = cv2.resize(image2, (150,150), interpolation=cv2.INTER_AREA)

        image_gray2 = self.grayify.transform(image_rs2)
        image_hog2 = self.hogify.transform([image_gray2])
        image_prepared2 = self.scalify.transform(image_hog2)

        pred_ver2 = self.sgd_clf_cup.predict(image_prepared2)
        print("pred", pred_ver2)

        if pred_ver2 == 1:
            pred_ver2_fin = self.sgd_clf_bb.predict(image_prepared2)
            print("prob pred", self.sgd_clf_bb.predict_proba(image_prepared2))
            print("Final Predction", pred_ver2_fin)

        image3 = cv2.imread(url3)
        image_rs3 = cv2.resize(image3, (150,150), interpolation=cv2.INTER_AREA)
    
        image_gray3 = self.grayify.transform(image_rs3)
        image_hog3 = self.hogify.transform([image_gray3])
        image_prepared3 = self.scalify.transform(image_hog3)

        pred_ver3 = self.sgd_clf_cup.predict(image_prepared3)
        print("pred", pred_ver3)

        if pred_ver3 == 1:
            pred_ver3_fin = self.sgd_clf_bb.predict(image_prepared3)
            print("prob pred", self.sgd_clf_bb.predict_proba(image_prepared3))
            print("Final Predction", pred_ver3_fin)

    def book_verify(self):
        path_books = "pic/val/test"
        flat_list = []
        for img in os.listdir(path_books):
            im = cv2.imread(os.path.join(path_books,img))
            im_rs = cv2.resize(im, (150,150), interpolation=cv2.INTER_AREA)
            flat_list.append(im_rs)
        flat_data = np.array(flat_list)

        data_gray = self.grayify.transform(flat_data)
        data_hog = self.hogify.transform(data_gray)
        data_prepared = self.scalify.transform(data_hog)
        lista_predizioni = []
        pred = self.sgd_clf_cup.predict(data_prepared)
        print("pred", pred)
        for pos,i in enumerate(pred):
            if i == 0:
                lista_predizioni.append('cup')
            elif i == 1:
                pred_new = self.sgd_clf_bb.predict([data_prepared[pos]])
                pred[pos] = pred_new
                if pred_new == 0:
                    lista_predizioni.append('book')
                elif pred_new == 1:
                    lista_predizioni.append('box')
        print("pred_fin", pred)
        print("lista_predizioni", lista_predizioni)


def mainhog():
    c = classifier()
    
    return c