'''
Created on Jan 14, 2015

@author: ckomurlu
'''

from abc import ABCMeta, abstractmethod
import numpy as np

# This is an abstract class, it's not meant to be
# instantiated, although the interpreter lets us do
# that, we should avoid creating MLModel instances
class MLModel(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
    
    @staticmethod
    def _dilate_mat(mat, label_vec, scheme_vec):
        if type(scheme_vec) is list:
            scheme_vec = np.array(scheme_vec)
        dilated_mat = np.zeros(shape=(mat.shape[0],scheme_vec.shape[0]))
        for i in range(label_vec.shape[0]):
            target_idx = np.where(label_vec[i] == scheme_vec)[0][0]
            dilated_mat[:,target_idx] = mat[:,i]
        return dilated_mat
    
    '''
    train_mat should be a two dimentional matrix composed of RandomVarNode
    '''
    @abstractmethod
    def fit(self, train_mat):
        pass
     
    '''
    Same as fit() function
    '''
    @abstractmethod
    def predict(self, test_mat, evid_mat, **kwargs): #return Y
        pass
    
    @abstractmethod
    def compute_accuracy(self, Y_test, Y_pred):
        pass

    @abstractmethod
    def compute_confusion_matrix(self, Y_test, Y_pred):
        pass


