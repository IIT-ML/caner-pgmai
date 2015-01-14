'''
Created on Jan 14, 2015

@author: ckomurlu
'''

from abc import ABCMeta, abstractmethod


# This is an abstract class, it's not meant to be
# instantiated, although the interpreter lets us do
# that, we should avoid creating MLModel instances
class MLModel(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self,X,Y):
        pass
     
    @abstractmethod
    def predict(self,X): #return Y
        pass
    
    @abstractmethod
    def compute_accuracy(self,Y_test,Y_pred):
        pass

    @abstractmethod
    def compute_confusion_matrix(self,Y_test,Y_pred):
        pass


