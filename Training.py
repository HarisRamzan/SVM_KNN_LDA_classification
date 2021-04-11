# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:28:39 2020

@author: haris
"""
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
from joblib import dump
from warnings import simplefilter
import numpy as np
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

def featurescaling(X_train,X_test):
    sc = MinMaxScaler(feature_range=[0, 1])
    #sc = StandardScaler()
    TrainX_std = sc.fit_transform(X_train)
    Testx_std = sc.fit_transform(X_test)
    return TrainX_std,Testx_std

def featureExtraction(TrainX_std,Testx_std):
    pca = decomposition.PCA(n_components=52)  #principle component analysis to feature extraction/reduction
    pca_standard_trainx = pca.fit_transform(TrainX_std)
    pca_standard_testx = pca.fit_transform(Testx_std)
    return pca_standard_trainx,pca_standard_testx

def TrainSVCModel(svc,pca_standard_trainx,y_train):
        svc.fit(pca_standard_trainx,y_train)
def Predictsvc(svc,pca_standard_testx):
    y_pred=svc.predict(pca_standard_testx)
    return y_pred
def calculatesvcAccuracy(y_test,prediction):
    print("svc accuracy:",accuracy_score(y_test,prediction))


def main():
    #reading csv files
    simplefilter(action='ignore', category=FutureWarning)
    X_train=np.loadtxt("TrainData.csv")
    y_train=np.loadtxt("TrainLabels.csv")
    X_test=np.loadtxt("TestData.csv")
    #y_test=np.loadtxt("")#actual label
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.20)
    
    TrainX_std,Testx_std=featurescaling(X_train,X_test);
    pca_standard_trainx,pca_standard_testx=featureExtraction(TrainX_std,Testx_std);
    
#    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
#    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
#    grid.fit(pca_standard_trainx,y_train)
#    print(grid.best_estimator_)
    
    #Training svc on basis of hyperparameters
    svc=SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    TrainSVCModel(svc,pca_standard_trainx,y_train);
    dump(svc, 'myModel.joblib');
    
    
if __name__ == '__main__':
        main()