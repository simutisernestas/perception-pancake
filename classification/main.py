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

# Helper function for plotting the fit of your SVM.
def plot_fit(X, y, clf):
    """
    X = samples
    y = Ground truth
    clf = trained model
    """
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig = plt.figure(1, figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors= "black")
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

def main():

    Categories=['box','cup','book']
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir='C:/Users/albor/OneDrive - Danmarks Tekniske Universitet/Dokumenter/PFAS/project/friendly-pancake/classification/pic/train' 
    #path which contains all the categories of images
    for i in Categories:
    
        print(f'loading... category : {i}')
        path=os.path.join(datadir,i).replace("\\","/")
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img)) #replace doesn't seem to work. Path is messed up .replace("\\","/")
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)
    df=pd.DataFrame(flat_data) #dataframe
    df['Target']=target
    x=df.iloc[:,:-1] #input data 
    y=df.iloc[:,-1] #output data
    
    print(x.shape())
    print(y.shape())
    
if __name__ == "__main__":
    main()
