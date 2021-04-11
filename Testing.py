# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:47:41 2020

@author: haris
"""

from joblib import load
import numpy as np
from warnings import simplefilter
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

def featurescaling(X_test):
    sc = MinMaxScaler(feature_range=[0, 1])
    #sc = StandardScaler()
    Testx_std = sc.fit_transform(X_test)
    return Testx_std

def featureExtraction(Testx_std):
    pca = decomposition.PCA(n_components=52)  #principle component analysis to feature extraction/reduction
    pca_standard_testx = pca.fit_transform(Testx_std)
    return pca_standard_testx

def Predictsvc(svc,pca_standard_testx):
    y_pred=svc.predict(pca_standard_testx)
    return y_pred

def main():
    #reading csv files
    simplefilter(action='ignore', category=FutureWarning)
    X_test=np.loadtxt("TestData.csv")
    
    Testx_std=featurescaling(X_test);
    pca_standard_testx=featureExtraction(Testx_std);
    
    clf = load('myModel.joblib');
    prediction=Predictsvc(clf,pca_standard_testx);
    np.savetxt("myPredictions.csv",prediction);
    
if __name__ == '__main__':
        main()