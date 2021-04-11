# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:25:14 2020

@author: haris
"""
import numpy as np
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
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

#LDA MODEL FUNCTIONS
def TrainLDAModel(LDA,pca_standard_trainx,y_train):
    LDA.fit(pca_standard_trainx,y_train)
    
def PredictLDA(LDA,pca_standard_testx):
    y_pred=LDA.predict(pca_standard_testx)
    return y_pred

def calculateLDAAccuracy(y_test,prediction):
    print("LDA accuracy:",accuracy_score(y_test,prediction))

#KNN MODEL FUNCTIONS

def TrainKNNModel(knn,pca_standard_trainx,y_train):
        knn.fit(pca_standard_trainx,y_train)
def Predictknn(knn,pca_standard_testx):
    y_pred=knn.predict(pca_standard_testx)
    return y_pred

def calculateknnAccuracy(y_test,prediction):
    print("knn accuracy:",accuracy_score(y_test,prediction))

def main():
    #reading csv files
    simplefilter(action='ignore', category=FutureWarning)
    X_train=np.loadtxt("TrainData.csv")
    y_train=np.loadtxt("TrainLabels.csv")
    X_test=np.loadtxt("TestData.csv")
    
    #X_train, X_test, y_train, y_test = train_test_split(Trainx,Trainy, test_size=0.20)
    
    TrainX_std,Testx_std=featurescaling(X_train,X_test);
    pca_standard_trainx,pca_standard_testx=featureExtraction(TrainX_std,Testx_std);
    
    
#    LDA = LinearDiscriminantAnalysis()
#    TrainLDAModel(LDA,pca_standard_trainx,y_train);
#    prediction=PredictLDA(LDA,pca_standard_testx);
#    calculateLDAAccuracy(y_test,prediction)
#    
#    ##Model2 KNN model training and accuracy
#    knn = KNeighborsClassifier(n_neighbors=5)
#    TrainKNNModel(knn,pca_standard_trainx,y_train);
#    prediction=Predictknn(knn,pca_standard_testx);
#    calculateknnAccuracy(y_test,prediction) 
    kf = KFold(n_splits=5)
    clf = SVC(kernel='linear', C=1)
    scores = cross_val_score(clf,pca_standard_trainx,y_train, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
    
    #Gridsearch is used to select best hyperparameters for svc
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    grid.fit(pca_standard_trainx,y_train)
    print(grid.best_estimator_)
    
    
    
if __name__ == '__main__':
        main()