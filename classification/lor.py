from tkinter import Grid
import fontTools
import mpl_toolkits
import os
import cv2

import sklearn as sk
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from skimage.transform import resize
from scipy.stats import uniform, truncnorm, randint
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier

import timeit
import time
import datetime as dt
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
import sklearn as sk
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import ssl

class classifier():
    def __init__(self):
        super().__init__()
        self.categories = ['box','cup','book']
        self.load_data()
        self.prepare_data()
        self.create_model()
        self.fit_model()
        self.test_model()

    def load_data(self):
        self.flat_data_arr = [] #input array
        self.target_arr = [] #output array
        
        ##Path to the training pics##
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

    def prepare_data(self):
        print("Data loaded, preparing it")
        self.start  =  timeit.default_timer()
        X_scaled  =  preprocessing.scale(self.flat_data)
        pca  =  PCA(n_components  =  2)
        principalComponents  =  pca.fit_transform(X_scaled)
        print(f'Principal components:{principalComponents[0]}')
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(principalComponents, self.targets, test_size = 0.4)
        print("Data prepared")

    def create_model(self):
        print("Creating the model")
        model_params = {
            # randomly sample numbers from 4 to 204 estimators
            'n_estimators': randint(4,200),
            # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
            'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
            # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
            'min_samples_split': uniform(0.01, 0.199)
        }

        # create random forest classifier model
        rf_model = RandomForestClassifier()

        # set up random search meta-estimator
        # this will train 100 models over 5 folds of cross validation (500 models total)
        self.clf = RandomizedSearchCV(rf_model, model_params, n_iter=100, cv=5, random_state=1)
        print("Model created")

    def fit_model(self):
        print("Fitting model")
        self.model = self.clf.fit(self.X_train, self.y_train)
        #best_par = model.best_params_
        print("Model fitted")

    def test_model(self):
        print("Testing model")
        self.model.score(self.X_train, self.y_train)
        self.model.score(self.X_test, self.y_test)
        pred  =  self.model.predict(self.X_test)
        
        stop  =  timeit.default_timer()    
        print(f'accuracy score:{accuracy_score(self.y_test, pred)}') 
        
        #Showing prediction
        print('Time: ', stop - self.start)
    
    def validate(self):
        print("Validation")
        ##èath to ONE pic to test and validate##
        path_validate = "pic\val\validation_cup.jpg"
        img_validate = imread(path_validate)
        img_resized_validate = resize(img_validate,(150,150,3))
        flat_data_validate = img_resized_validate.flatten()
        probability=self.model.predict_proba(flat_data_validate)
        for ind,val in enumerate(self.categories):
            print(f'{val} = {probability[0][ind]*100}%')
        print("The predicted image is : "+self.categories[self.model.predict(flat_data_validate)[0]])
        

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
