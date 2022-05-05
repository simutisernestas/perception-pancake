from tkinter import Grid
import fontTools
import mpl_toolkits
import os
import cv2

import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imread

import timeit
import time
import datetime as dt
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import ssl

class classifier():
    def __init__(self):
        super().__init__()
        self.categories = ['box','cup','book']
        self.load_data()
        self.create_model()

    def load_data(self):
        self.flat_data_arr = [] #input array
        self.target_arr = [] #output array
        self.datadir = 'pic/train' 
        #path which contains all the categories of images
        for i in self.categories:
        
            print(f'loading... category : {i}')
            path = os.path.join(self.datadir,i).replace("\\","/")
            for img in os.listdir(path):
                img_array = imread(os.path.join(path,img))
                img_resized = resize(img_array,(150,150,3))
                self.flat_data_arr.append(img_resized.flatten())
                self.target_arr.append(self.categories.index(i))
            print(f'loaded category:{i} successfully')
        self.flat_data = np.array(self.flat_data_arr)
        self.targets = np.array(self.target_arr)

    def create_model(self):
    ##  Can convert to Pandas
        # df = pd.DataFrame(flat_data) #dataframe
        # df['Target'] = target
        # x = df.iloc[:,:-1] #input data 
        # y = df.iloc[:,-1] #output data
        # print(np.shape(x))
        # print(np.shape(y))
        self.start  =  timeit.default_timer()
        X_scaled  =  StandardScaler().fit_transform(self.flat_data)
        pca  =  PCA(n_components  =  2)
        principalComponents  =  pca.fit_transform(X_scaled)
        print(f'Principal components:{principalComponents[0]}')
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(principalComponents, self.targets, test_size = 0.4)

        param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}

        svc = svm.SVC(probability=True)
        model = GridSearchCV(svc, param_grid)

               
        model.fit(self.X_train, self.y_train)
        best_par = model.best_params_
        model.score(self.X_train, self.y_train)
        model.score(self.X_test, self.y_test)
        pred  =  model.predict(self.X_test)
        
        stop  =  timeit.default_timer()    
        print(f'accuracy score:{accuracy_score(self.y_test, pred)}') 
        print("best params", best_par)
        #Showing prediction
        print('Time: ', stop - self.start)

    # Helper function for plotting the fit of your SVM.
    def plot_fit(self, X, y):
        """
        X  =  samples
        y  =  Ground truth
        clf  =  trained model
        """
        h  =  .02
        x_min, x_max  =  X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max  =  X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy  =  np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Z  =  self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z  =  Z.reshape(xx.shape)
        fig  =  plt.figure(1, figsize = (8, 6))
        plt.contourf(xx, yy, Z, cmap = plt.cm.coolwarm, alpha = 0.8)
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.coolwarm, edgecolors =  "black")
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())




    
if __name__  ==  "__main__":
    
    cat_num = [1,2,3] #to return to the main function file, to distinguish between the categories
    
    mod = classifier()
