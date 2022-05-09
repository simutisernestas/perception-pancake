import os
import numpy as np
import pandas as pd
import timeit
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import cv2

###############
#TO CALL THIS FUNCTION, JUST USE THIS:
"""def load_classifier():

    classer = mainno()

    mod_cup = classer.model_cup
    mod_bb = classer.model_bb

    return mod_cup, mod_bb"""
#THEN USE "mod.predict([img])[0]" TO PREDICT WHETHER IT IS A CUP OR ELSE (0 OR 1), 
#THEN "mod.predict([img])[0]" TO PREDICT WHETHER IT IS A BOX OR A BOOK (0 OR 1)
###############


class classifier():
    def __init__(self):
        super().__init__()
        
        self.cup_model()
        self.bb_model()
        #self.verify()
        #self.book_verify()

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
                img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                flat_data_arr.append(img_gray.flatten())
                target_arr.append(categories.index(i))
            print(f'loaded category:{i} successfully')
        #print("flat_list", len(flat_data_arr))
        flat_data = np.array(flat_data_arr)
        #print("flat_size", flat_data.shape)
        targets = np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        df['Target']=targets
        df_shuffled = shuffle(df) 
        x=df_shuffled.iloc[:,:-1] #input data 
        y=df_shuffled.iloc[:,-1] #output data
        
        start  =  timeit.default_timer()

        X_train, X_test, y_train, y_test =  train_test_split(x, y, test_size = 0.4)
        #print("xtesto", X_test.shape)

        svm  =  SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        self.model_cup = svm.fit(X_train, y_train)

        
        self.model_cup.score(X_train, y_train)
        self.model_cup.score(X_test, y_test)
        pred  =  self.model_cup.predict(X_test)
        
        stop  =  timeit.default_timer()    
        print(f'accuracy score:{accuracy_score(y_test, pred)}') 
        #Showing prediction
        print('Time: ', stop - start)

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
                img_res = cv2.resize(img_array, (150,150), interpolation=cv2.INTER_AREA)
                img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                flat_data_arr.append(img_gray.flatten())
                target_arr.append(categories.index(i))
            print(f'loaded category:{i} successfully')
        #print("flat_list", len(flat_data_arr))
        flat_data = np.array(flat_data_arr)
        #print("flat_size", flat_data.shape)
        targets = np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        df['Target']=targets
        df_shuffled = shuffle(df) 
        x=df_shuffled.iloc[:,:-1] #input data 
        y=df_shuffled.iloc[:,-1] #output data

        start  =  timeit.default_timer()

        X_train, X_test, y_train, y_test =  train_test_split(x, y, test_size = 0.4)


        svm  =  SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        self.model_bb = svm.fit(X_train, y_train)

        self.model_bb.score(X_train, y_train)
        self.model_bb.score(X_test, y_test)
        pred  =  self.model_bb.predict(X_test)
       
        stop  =  timeit.default_timer()    
        print(f'accuracy score:{accuracy_score(y_test, pred)}') 
        #Showing prediction
        print('Time: ', stop - start)


    def verify(self):
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
        print("lista imflat", [image_flat])

        pred_ver1 = self.model_cup.predict([image_flat])
        print("pred1", pred_ver1)
        #print("proba pred1", self.model_cup.predict_proba([image_flat]))

        if pred_ver1 == 1:
            pred_ver1_fin = self.model_bb.predict([image_flat])
            print("pred 1 fin", pred_ver1_fin)
            #print("proba pred1 fin", self.model_bb.predict_proba([image_flat]))


        image2 = cv2.imread(url2)
        image_rs2 = cv2.resize(image2, (150,150), interpolation=cv2.INTER_AREA)
        image_flat2 = image_rs2.flatten()
        #print("a lista",[image_flat])
        pred_ver2 = self.model_cup.predict([image_flat2])
        print("pred2", pred_ver2)
        #print("proba pred2", self.model_cup.predict_proba([image_flat2]))

        if pred_ver2 == 1:
            pred_ver2_fin = self.model_bb.predict([image_flat2])
            print("pred 2 fin", pred_ver2_fin)
            #print("proba pred2 fin", self.model_bb.predict_proba([image_flat2]))


        image3 = cv2.imread(url3)
        image_rs3 = cv2.resize(image3, (150,150), interpolation=cv2.INTER_AREA)
        image_flat3 = image_rs3.flatten()
        #imlist.append(image_flat3)
        pred_ver3 = self.model_cup.predict([image_flat3])
        print("pred3", pred_ver3)
        #print("proba pred3", self.model_cup.predict_proba([image_flat3]))
        if pred_ver3 == 1:
            pred_ver3_fin = self.model_bb.predict([image_flat3])
            print("pred 3 fin", pred_ver3_fin)
            #print("proba pred3 fin", self.model_bb.predict_proba([image_flat3]))

    def book_verify(self):
        path_books = "pic/val/bookcup"
        flat_list = []
        for img in os.listdir(path_books):
            im = cv2.imread(os.path.join(path_books,img))
            im_rs = cv2.resize(im, (150,150), interpolation=cv2.INTER_AREA)
            im_flat = im_rs.flatten()
            flat_list.append(im_flat)
        flat_data = np.array(flat_list)

        pred = self.model_cup.predict(flat_data)
        #pred_prob = self.model_cup.predict_proba(flat_data)
        #print("pred proba", pred_prob)
        print("pred", pred)
        for pos,i in enumerate(pred):
            if i == 1:
                print("di nuovo", pos)
                pred_new = self.model_bb.predict([flat_data[pos]])
                pred[pos] = pred_new
        print("pred_fin", pred)

def mainno():
    c = classifier()
    
    return c