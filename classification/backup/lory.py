import os
from cv2 import INTER_AREA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import skimage.transform
from sklearn.calibration import CalibratedClassifierCV
from skimage.io import imread
import timeit
from sklearn.decomposition import PCA
from sklearn import datasets, svm
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, truncnorm, randint
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import cv2
import pickle
from joblib import dump, load


class classifier():
    def __init__(self):
        super().__init__()
        
        self.cup_model()
        #self.bb_model()
        self.verify()

    def cup_model(self):
        categories = ['cup','box']
        flat_data_arr = [] #input array
        target_arr = [] #output array
        datadir = 'pic/train' 
        #path which contains all the categories of images
        for i in categories:
            print(f'loading... category : {i}')
            path = os.path.join(datadir,i).replace("\\","/")
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img))

                #img_class = cv2.imread(path_class)
                #print("img_class", img_array.shape)
                """plt.imshow(img_array)
                plt.show()"""
                img_res = cv2.resize(img_array, (150,150), interpolation=cv2.INTER_AREA)
                #print("img_Res", img_res.shape)
                """plt.imshow(img_res)
                plt.show()"""
                #img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                #print("img_gray", img_gray.shape)
                """plt.imshow(img_gray, cmap = "gray")
                plt.show()"""

                flat_data_arr.append(img_res.flatten())
                target_arr.append(categories.index(i))
            print(f'loaded category:{i} successfully')
        print("flat_list", len(flat_data_arr))
        flat_data = np.array(flat_data_arr)
        print("flat_size", flat_data.shape)
        targets = np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        df['Target']=targets
        df_shuffled = shuffle(df) 
        x=df_shuffled.iloc[:,:-1] #input data 
        y=df_shuffled.iloc[:,-1] #output data
        
        start  =  timeit.default_timer()

        X_scaled  =  StandardScaler().fit_transform(flat_data)
        self.pca_cup  =  PCA(n_components  =  20)

        #dump(self.pca_cup,'pca_cup.joblib')

        self.principalComponents  =  self.pca_cup.fit_transform(X_scaled)
        print(f'Principal components:{self.principalComponents[0]}')
        print(self.pca_cup.explained_variance_ratio_)
        print(self.pca_cup.explained_variance_)
        X_train, X_test, y_train, y_test =  train_test_split(self.principalComponents, targets, test_size = 0.4)
        #X_train, X_test, y_train, y_test =  train_test_split(flat_data, targets, test_size = 0.4)
        print("xtesto", X_test.shape)
        svm  =  LinearSVC(penalty = 'l2', loss = 'squared_hinge', random_state = 0, max_iter = 10e4)
        clf = CalibratedClassifierCV(svm) 
        self.model_cup = clf.fit(X_train, y_train)

        """model_params = {
        # randomly sample numbers from 4 to 204 estimators
        'n_estimators': randint(4,200),
        # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
        'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
        # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
        'min_samples_split': uniform(0.01, 0.199)
        }"""

        #rf_model_cup = RandomForestClassifier()
        #self.model_cup = RandomizedSearchCV(rf_model_cup, model_params, n_iter=100, cv=5, random_state=1)


        #self.model_cup.fit(X_train,y_train)

        self.model_cup.score(X_train, y_train)
        self.model_cup.score(X_test, y_test)
        pred  =  self.model_cup.predict(X_test)
        print("first pred", pred)
        print("ytest", y_test)
        
        stop  =  timeit.default_timer()    
        print(f'accuracy score:{accuracy_score(y_test, pred)}') 
        #Showing prediction
        print('Time: ', stop - start)

        #dump(self.model_cup, 'model_cup.joblib') 



    def bb_model(self):
        categories = ['box','book']
        flat_data_arr = [] #input array
        target_arr = [] #output array
        datadir = 'pic/train' 
        #path which contains all the categories of images
        for i in categories:
            print(f'loading... category : {i}')
            path = os.path.join(datadir,i).replace("\\","/")
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img))

                #img_class = cv2.imread(path_class)
                #print("img_class", img_array.shape)
                """plt.imshow(img_array)
                plt.show()"""
                img_res = cv2.resize(img_array, (150,150), interpolation=cv2.INTER_AREA)
                #print("img_Res", img_res.shape)
                """plt.imshow(img_res)
                plt.show()"""
                #img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                #print("img_gray", img_gray.shape)
                """plt.imshow(img_gray, cmap = "gray")
                plt.show()"""
                
                flat_data_arr.append(img_res.flatten())
                target_arr.append(categories.index(i))
            print(f'loaded category:{i} successfully')
        print("flat_list", len(flat_data_arr))
        flat_data = np.array(flat_data_arr)
        print("flat_size", flat_data.shape)
        targets = np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        df['Target']=targets
        df_shuffled = shuffle(df) 
        x=df_shuffled.iloc[:,:-1] #input data 
        y=df_shuffled.iloc[:,-1] #output data

        start  =  timeit.default_timer()

        #X_scaled  =  StandardScaler().fit_transform(flat_data)
        #self.pca_bb  =  PCA(n_components  =  20)

        #dump(self.pca_bb,'pca_bb.joblib')

        #self.principalComponents_bb  =  self.pca_bb.fit_transform(X_scaled)
        #print(f'Principal components:{self.principalComponents_bb[0]}')
        #print(self.pca_bb.explained_variance_ratio_)
        #print(self.pca_bb.explained_variance_)
        #X_train, X_test, y_train, y_test =  train_test_split(self.principalComponents_bb, targets, test_size = 0.4)
        X_train, X_test, y_train, y_test =  train_test_split(flat_data, targets, test_size = 0.4)


        svm  =  LinearSVC(penalty = 'l2', loss = 'squared_hinge', random_state = 0, max_iter = 10e4)
        clf = CalibratedClassifierCV(svm) 
        self.model_bb = clf.fit(X_train, y_train)

        """model_params = {
        # randomly sample numbers from 4 to 204 estimators
        'n_estimators': randint(4,200),
        # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
        'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
        # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
        'min_samples_split': uniform(0.01, 0.199)
        }

        rf_model = RandomForestClassifier()
        self.model = RandomizedSearchCV(rf_model, model_params, n_iter=100, cv=5, random_state=1)


        self.model.fit(X_train,y_train)"""


        self.model_bb.score(X_train, y_train)
        self.model_bb.score(X_test, y_test)
        pred  =  self.model_bb.predict(X_test)
        print("first pred", pred)
        stop  =  timeit.default_timer()    
        print(f'accuracy score:{accuracy_score(y_test, pred)}') 
        #Showing prediction
        print('Time: ', stop - start)

        #dump(self.model_bb, 'bb_model.joblib') 

    def verify(self):
        imlist = []
        ##LOAD IMAGE, RESIZE, GRAYSCALE AND FLATTEN IT##
        url = "pic/val/validation_book.jpg"
        url2 = "pic/val/validation_cup.jpg"
        url3 = "pic/val/validation_box.jpg"


        image = cv2.imread(url)
        image_rs = cv2.resize(image, (150,150), interpolation=cv2.INTER_AREA)
        #image_gray = cv2.cvtColor(image_rs, cv2.COLOR_BGR2GRAY)
        image_flat = image_rs.flatten()
        #print("image_gray", image_gray.shape)
        print("image_rs", image_rs.shape)
        print("image_flat", image_flat.shape)
        print("len imflat", len([image_flat]))
        
        #X_scaled_ver  =  StandardScaler().fit_transform([image_rs])
        pca_ver  =  PCA(n_components  =  20)

        #dump(self.pca_cup,'pca_cup.joblib')
        
        principalComponents_ver1  =  pca_ver.fit_transform([image_flat])

        #imlist.append(image_flat)
        
        print("princ_size", principalComponents_ver1.shape)
        print(f'Principal components:{principalComponents_ver1[0]}')
        pred_ver1 = self.model_cup.predict(principalComponents_ver1)
        print("pred", pred_ver1)


        image2 = cv2.imread(url2)
        image_rs2 = cv2.resize(image2, (150,150), interpolation=cv2.INTER_AREA)
        image_flat2 = image_rs2.flatten()
        imlist.append(image_flat2)
        #print("a lista",[image_flat])
        principalComponents_ver2 = self.pca_cup.transform([image_flat])
        print("princ_size", principalComponents_ver2.shape)
        print(f'Principal components:{principalComponents_ver2[0]}')
        pred_ver2 = self.model_cup.predict(principalComponents_ver2)
        print("pred", pred_ver2)


        image3 = cv2.imread(url3)
        image_rs3 = cv2.resize(image3, (150,150), interpolation=cv2.INTER_AREA)
        image_flat3 = image_rs3.flatten()
        #imlist.append(image_flat3)
        principalComponents_ver3 = self.pca_cup.transform([image_flat])
        print("princ_size", principalComponents_ver3.shape)
        print(f'Principal components:{principalComponents_ver3[0]}')
        pred_ver3 = self.model_cup.predict(principalComponents_ver3)
        print("pred", pred_ver3)



#def mainno():
if __name__ == "__main__":
    c = classifier()
    
    #return c
    
    #picca = c.pca_cup
    #mod_cup= c.model_cup
    #pbb = c.pca_bb
    #mod_bb = c.model
    #print(picca)

    #return picca, mod_cup, pbb, mod_bb

"""path_verify = "C:/Users/lorec/Desktop/LAgit/friendly-pancake/classification/pic/val/validation_book.jpg"
img_ver = imread(path_verify)
img_resized_ver = resize(img_ver, (150,150,3))
print("img_res_size", img_resized_ver.shape)
img_flat_ver = img_resized_ver.flatten()
principalComponents_ver = pca.transform([img_flat_ver])
print("princ_size", principalComponents_ver.shape)
print(f'Principal components:{principalComponents_ver[0]}')
pred_ver = model.predict(principalComponents_ver)
print("pred", pred_ver)"""


