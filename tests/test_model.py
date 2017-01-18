'''
Created on Jan 14, 2015

@author: ckomurlu
'''

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from models.ml_model import MLModel

class TestModel(MLModel):
    
    def __init__(self):
        self.lr_model = LogisticRegression()
    
    def fit(self,X_train,Y_train):
        self.lr_model.fit(X_train, Y_train)
    
    def predict(self, X):
        return self.lr_model.predict(X)

    def compute_accuracy(self,Y_test,Y_pred):
        return accuracy_score(Y_test,Y_pred)
    
    def compute_confusion_matrix(self, Y_test, Y_pred, labels=None):
        return confusion_matrix(Y_test, Y_pred, labels)






print 'test_model started'
ml_model = TestModel()
X = np.array([[0,1],[1,1],[1,0]])
Y = np.array([0,1,0])
ml_model.fit(X, Y)
X_test = np.array([[0,0]])
Y_pred = ml_model.predict(X_test)
Y_test = np.array([0])
print ml_model.compute_confusion_matrix(Y_test, Y_pred, [0,1])
print 'test_model ended'