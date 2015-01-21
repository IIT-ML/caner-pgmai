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
    
    '''
    train_mat should be a two dimentional matrix composed of RandomVarNode
    '''
    @abstractmethod
    def fit(self,train_mat):
        pass
     
    '''
    Same as fit() function
    '''
    @abstractmethod
    def predict(self,test_mat): #return Y
        pass
    
    @abstractmethod
    def compute_accuracy(self,Y_test,Y_pred):
        pass

    @abstractmethod
    def compute_confusion_matrix(self,Y_test,Y_pred):
        pass


